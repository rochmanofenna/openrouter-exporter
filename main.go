package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/reski/openrouter-exporter/cache"
	"github.com/reski/openrouter-exporter/client"
	"github.com/reski/openrouter-exporter/collector"
	"github.com/reski/openrouter-exporter/config"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	apiClient := client.NewClient(cfg.BaseURL, cfg.APIKey, cfg.APITimeout, cfg.MaxConcurrency)
	c := cache.New(apiClient, cfg.ScrapeInterval, logger)

	if (len(cfg.ActivityModels) > 0 || len(cfg.ActivityProviders) > 0) && cfg.ActivitySessionCookie != "" {
		c.SetActivityConfig(cfg.ActivityModels, cfg.ActivitySessionCookie, cfg.ActivityScrapeInterval)
		c.SetProviderActivity(cfg.ActivityProviders)
		logger.Info("activity scraping configured",
			"models", cfg.ActivityModels,
			"providers", cfg.ActivityProviders,
			"interval", cfg.ActivityScrapeInterval,
		)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	logger.Info("starting initial fetch")
	if err := c.Start(ctx); err != nil {
		logger.Error("initial fetch failed", "error", err)
		os.Exit(1)
	}

	prometheus.MustRegister(collector.New(c, logger))

	mux := http.NewServeMux()
	mux.Handle(cfg.MetricsPath, promhttp.Handler())
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if c.Get() != nil {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("ok"))
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, `<html><body><h1>OpenRouter Exporter</h1><p><a href="%s">Metrics</a></p></body></html>`, cfg.MetricsPath)
	})

	server := &http.Server{
		Addr:    cfg.ListenAddress,
		Handler: mux,
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		logger.Info("shutting down")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer shutdownCancel()
		server.Shutdown(shutdownCtx)
		c.Stop()
		cancel()
	}()

	logger.Info("listening", "address", cfg.ListenAddress, "metrics", cfg.MetricsPath)
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		logger.Error("server error", "error", err)
		os.Exit(1)
	}
}
