package collector

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/reski/openrouter-exporter/cache"
)

type OpenRouterCollector struct {
	cache  *cache.Cache
	logger *slog.Logger
}

func New(c *cache.Cache, logger *slog.Logger) *OpenRouterCollector {
	return &OpenRouterCollector{cache: c, logger: logger}
}

func (c *OpenRouterCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- MetricInputPrice
	ch <- MetricOutputPrice
	ch <- MetricCacheReadPrice
	ch <- MetricUptime30m
	ch <- MetricUptime5m
	ch <- MetricUptime1d
	ch <- MetricThroughput
	ch <- MetricLatency
	ch <- MetricModelInfo
	ch <- MetricScrapeDuration
	ch <- MetricScrapeErrors
	ch <- MetricModelsScraped
	ch <- MetricEndpointsScraped
	ch <- MetricScrapeTimestamp
	ch <- MetricActivityRequests
	ch <- MetricActivityPromptTokens
	ch <- MetricActivityOutputTokens
	ch <- MetricActivityCompletionTokens
	ch <- MetricActivityCacheHitTokens
	ch <- MetricActivityReasoningTokens
	ch <- MetricActivityInputOutputRatio
	ch <- MetricActivityEstCostDollars
	ch <- MetricActivityScrapeDuration
	ch <- MetricActivityScrapeErrors
	ch <- MetricActivityScrapeTimestamp
	ch <- MetricActivityPromptTokens5mDelta
}

func (c *OpenRouterCollector) Collect(ch chan<- prometheus.Metric) {
	data := c.cache.Get()
	if data == nil {
		return
	}

	// Scrape metadata
	ch <- prometheus.MustNewConstMetric(MetricScrapeDuration, prometheus.GaugeValue, data.FetchDuration.Seconds())
	ch <- prometheus.MustNewConstMetric(MetricScrapeErrors, prometheus.GaugeValue, float64(data.FetchErrors))
	ch <- prometheus.MustNewConstMetric(MetricModelsScraped, prometheus.GaugeValue, float64(data.ModelsCount))
	ch <- prometheus.MustNewConstMetric(MetricEndpointsScraped, prometheus.GaugeValue, float64(data.EndpointsCount))
	ch <- prometheus.MustNewConstMetric(MetricScrapeTimestamp, prometheus.GaugeValue, float64(data.FetchedAt.Unix()))

	// Per-model info metrics
	for _, m := range data.Models {
		modality := m.Architecture.Modality
		tokenizer := m.Architecture.Tokenizer
		inputMod := strings.Join(m.Architecture.InputModalities, ",")
		outputMod := strings.Join(m.Architecture.OutputModalities, ",")

		ch <- prometheus.MustNewConstMetric(
			MetricModelInfo, prometheus.GaugeValue, 1,
			m.ID, m.Name, strconv.Itoa(m.ContextLength),
			modality, tokenizer, inputMod, outputMod,
		)
	}

	// Per-endpoint metrics - deduplicate by tag across models
	seen := make(map[string]bool)
	for modelID, epResp := range data.Endpoints {
		for _, ep := range epResp.Data.Endpoints {
			key := modelID + "/" + ep.Tag
			if seen[key] {
				continue
			}
			seen[key] = true

			c.emitEndpointMetrics(ch, modelID, ep)
		}
	}

	// Activity metrics - emit only the latest date per model
	if data.Activity != nil {
		ch <- prometheus.MustNewConstMetric(MetricActivityScrapeErrors, prometheus.GaugeValue, float64(data.ActivityErrors))

		for modelID, records := range data.Activity {
			if len(records) == 0 {
				continue
			}
			r := records[0] // analytics records are sorted newest-first
			date := strings.SplitN(r.Date, " ", 2)[0]

			outputTokens := r.TotalCompletionTokens
			completionTokens := outputTokens - r.TotalNativeTokensReasoning
			if completionTokens < 0 {
				completionTokens = 0
			}

			ch <- prometheus.MustNewConstMetric(MetricActivityRequests, prometheus.GaugeValue, float64(r.Count), modelID, date)
			ch <- prometheus.MustNewConstMetric(MetricActivityPromptTokens, prometheus.GaugeValue, float64(r.TotalPromptTokens), modelID, date)
			ch <- prometheus.MustNewConstMetric(MetricActivityOutputTokens, prometheus.GaugeValue, float64(outputTokens), modelID, date)
			ch <- prometheus.MustNewConstMetric(MetricActivityCompletionTokens, prometheus.GaugeValue, float64(completionTokens), modelID, date)
			ch <- prometheus.MustNewConstMetric(MetricActivityCacheHitTokens, prometheus.GaugeValue, float64(r.TotalNativeTokensCached), modelID, date)
			ch <- prometheus.MustNewConstMetric(MetricActivityReasoningTokens, prometheus.GaugeValue, float64(r.TotalNativeTokensReasoning), modelID, date)

			if outputTokens > 0 {
				ratio := float64(r.TotalPromptTokens) / float64(outputTokens)
				ch <- prometheus.MustNewConstMetric(MetricActivityInputOutputRatio, prometheus.GaugeValue, ratio, modelID, date)
			}

			if cost, ok := c.estCost(data, modelID, r); ok {
				ch <- prometheus.MustNewConstMetric(MetricActivityEstCostDollars, prometheus.GaugeValue, cost, modelID, date)
			}
		}
	}

	for modelID, delta := range data.PromptTokenDeltas {
		ch <- prometheus.MustNewConstMetric(
			MetricActivityPromptTokens5mDelta,
			prometheus.GaugeValue,
			float64(delta),
			modelID,
		)
	}
}

func (c *OpenRouterCollector) emitEndpointMetrics(ch chan<- prometheus.Metric, modelID string, ep cache.Endpoint) {
	provider := ep.ProviderName
	tag := ep.Tag
	quant := ep.Quantization

	// Pricing
	if price, err := parsePrice(ep.Pricing.Input); err == nil {
		ch <- prometheus.MustNewConstMetric(MetricInputPrice, prometheus.GaugeValue, price, modelID, provider, tag, quant)
	} else {
		c.logger.Warn("invalid input price", "model", modelID, "provider", provider, "value", ep.Pricing.Input, "error", err)
	}

	if price, err := parsePrice(ep.Pricing.Output); err == nil {
		ch <- prometheus.MustNewConstMetric(MetricOutputPrice, prometheus.GaugeValue, price, modelID, provider, tag, quant)
	} else {
		c.logger.Warn("invalid output price", "model", modelID, "provider", provider, "value", ep.Pricing.Output, "error", err)
	}

	if ep.Pricing.InputCacheRead != nil {
		if price, err := parsePrice(*ep.Pricing.InputCacheRead); err == nil {
			ch <- prometheus.MustNewConstMetric(MetricCacheReadPrice, prometheus.GaugeValue, price, modelID, provider, tag, quant)
		}
	}

	// Uptime
	ch <- prometheus.MustNewConstMetric(MetricUptime5m, prometheus.GaugeValue, ep.UptimeLast5m, modelID, provider, tag, quant)
	ch <- prometheus.MustNewConstMetric(MetricUptime30m, prometheus.GaugeValue, ep.UptimeLast30m, modelID, provider, tag, quant)
	ch <- prometheus.MustNewConstMetric(MetricUptime1d, prometheus.GaugeValue, ep.UptimeLast1d, modelID, provider, tag, quant)

	// Throughput (auth-only)
	if ep.ThroughputLast30m != nil {
		ch <- prometheus.MustNewConstMetric(MetricThroughput, prometheus.GaugeValue, ep.ThroughputLast30m.P50, modelID, provider, tag, quant, "p50")
		ch <- prometheus.MustNewConstMetric(MetricThroughput, prometheus.GaugeValue, ep.ThroughputLast30m.P75, modelID, provider, tag, quant, "p75")
		ch <- prometheus.MustNewConstMetric(MetricThroughput, prometheus.GaugeValue, ep.ThroughputLast30m.P90, modelID, provider, tag, quant, "p90")
		ch <- prometheus.MustNewConstMetric(MetricThroughput, prometheus.GaugeValue, ep.ThroughputLast30m.P99, modelID, provider, tag, quant, "p99")
	}

	// Latency (auth-only)
	if ep.LatencyLast30m != nil {
		ch <- prometheus.MustNewConstMetric(MetricLatency, prometheus.GaugeValue, ep.LatencyLast30m.P50*1000, modelID, provider, tag, quant, "p50")
		ch <- prometheus.MustNewConstMetric(MetricLatency, prometheus.GaugeValue, ep.LatencyLast30m.P75*1000, modelID, provider, tag, quant, "p75")
		ch <- prometheus.MustNewConstMetric(MetricLatency, prometheus.GaugeValue, ep.LatencyLast30m.P90*1000, modelID, provider, tag, quant, "p90")
		ch <- prometheus.MustNewConstMetric(MetricLatency, prometheus.GaugeValue, ep.LatencyLast30m.P99*1000, modelID, provider, tag, quant, "p99")
	}
}

func (c *OpenRouterCollector) estCost(data *cache.CachedData, modelID string, r cache.ActivityRecord) (float64, bool) {
	epResp, ok := data.Endpoints[modelID]
	if !ok || len(epResp.Data.Endpoints) == 0 {
		return 0, false
	}
	ep := epResp.Data.Endpoints[0]

	inputPrice, err := strconv.ParseFloat(ep.Pricing.Input, 64)
	if err != nil {
		return 0, false
	}
	outputPrice, err := strconv.ParseFloat(ep.Pricing.Output, 64)
	if err != nil {
		return 0, false
	}

	// Prices are in USD per million tokens
	cost := float64(r.TotalPromptTokens)*inputPrice/1_000_000 +
		float64(r.TotalCompletionTokens)*outputPrice/1_000_000
	return cost, true
}

func parsePrice(s string) (float64, error) {
	if s == "" {
		return 0, fmt.Errorf("empty price")
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parse price %q: %w", s, err)
	}
	return v * 1_000_000, nil
}
