package cache

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/reski/openrouter-exporter/client"
)

type CachedData struct {
	Models         []Model
	Endpoints      map[string]*EndpointsResponse
	Activity       map[string][]ActivityRecord
	FetchedAt      time.Time
	FetchDuration  time.Duration
	FetchErrors    int
	ModelsCount    int
	EndpointsCount int
	ActivityErrors int
	PromptTokenDeltas map[string]int64
}

// Re-export client types for cache consumers
type Model = client.Model
type EndpointsResponse = client.EndpointsResponse
type Endpoint = client.Endpoint
type ActivityRecord = client.ActivityRecord

type Cache struct {
	mu                    sync.RWMutex
	data                  *CachedData
	client                *client.OpenRouterClient
	interval              time.Duration
	activityInterval      time.Duration
	stopCh                chan struct{}
	logger                *slog.Logger
	activityModels        []string
	sessionCookie         string
	prevPromptTokens      map[string]int64
}

func New(c *client.OpenRouterClient, interval time.Duration, logger *slog.Logger) *Cache {
	return &Cache{
		client:   c,
		interval: interval,
		stopCh:   make(chan struct{}),
		logger:   logger,
	}
}

func (c *Cache) SetActivityConfig(models []string, sessionCookie string, interval time.Duration) {
	c.activityModels = models
	c.sessionCookie = sessionCookie
	c.activityInterval = interval
}

func (c *Cache) Start(ctx context.Context) error {
	if err := c.refresh(ctx); err != nil {
		return err
	}

	go c.run(ctx)
	return nil
}

func (c *Cache) Get() *CachedData {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.data
}

func (c *Cache) Stop() {
	close(c.stopCh)
}

func (c *Cache) run(ctx context.Context) {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	var activityTicker *time.Ticker
	var activityCh <-chan time.Time
	if len(c.activityModels) > 0 && c.sessionCookie != "" {
		activityTicker = time.NewTicker(c.activityInterval)
		activityCh = activityTicker.C
		defer activityTicker.Stop()
	}

	for {
		select {
		case <-ticker.C:
			if err := c.refresh(ctx); err != nil {
				c.logger.Error("cache refresh failed", "error", err)
			}
		case <-activityCh:
			c.refreshActivity(ctx)
		case <-c.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

func (c *Cache) refresh(ctx context.Context) error {
	start := time.Now()

	modelsResp, err := c.client.FetchModels(ctx)
	if err != nil {
		return fmt.Errorf("fetch models: %w", err)
	}
	models := modelsResp.Data

	result, err := c.client.FetchAllEndpoints(ctx, models)
	if err != nil {
		return fmt.Errorf("fetch endpoints: %w", err)
	}

	// Count total endpoints
	totalEndpoints := 0
	for _, ep := range result.Endpoints {
		totalEndpoints += len(ep.Data.Endpoints)
	}

	cached := &CachedData{
		Models:         models,
		Endpoints:      result.Endpoints,
		FetchedAt:      time.Now(),
		FetchDuration:  time.Since(start),
		FetchErrors:    result.Errors,
		ModelsCount:    len(models),
		EndpointsCount: totalEndpoints,
	}

	// Fetch activity data if configured
	var deltas, newPrev map[string]int64
	if len(c.activityModels) > 0 && c.sessionCookie != "" {
		activityResult, err := c.client.FetchAllActivity(ctx, c.activityModels, c.sessionCookie)
		if err != nil {
			c.logger.Error("fetch activity failed", "error", err)
		} else {
			cached.Activity = activityResult.Activity
			cached.ActivityErrors = activityResult.Errors
			deltas, newPrev = computePromptTokenDeltas(activityResult.Activity, c.prevPromptTokens)
			cached.PromptTokenDeltas = deltas
		}
	}

	c.mu.Lock()
	c.data = cached
	if newPrev != nil {
		c.prevPromptTokens = newPrev
	}
	c.mu.Unlock()

	c.logger.Info("cache refreshed",
		"models", cached.ModelsCount,
		"endpoints", cached.EndpointsCount,
		"errors", cached.FetchErrors,
		"duration", cached.FetchDuration.Round(time.Millisecond),
	)

	return nil
}

func (c *Cache) refreshActivity(ctx context.Context) {
	if len(c.activityModels) == 0 || c.sessionCookie == "" {
		return
	}

	activityResult, err := c.client.FetchAllActivity(ctx, c.activityModels, c.sessionCookie)
	if err != nil {
		c.logger.Error("activity refresh failed", "error", err)
		return
	}

	deltas, newPrev := computePromptTokenDeltas(activityResult.Activity, c.prevPromptTokens)

	c.mu.Lock()
	if c.data != nil {
		c.data.Activity = activityResult.Activity
		c.data.ActivityErrors = activityResult.Errors
		c.data.PromptTokenDeltas = deltas
	}
	c.prevPromptTokens = newPrev
	c.mu.Unlock()

	c.logger.Info("activity refreshed",
		"models", len(activityResult.Activity),
		"errors", activityResult.Errors,
	)
}

// latestDayPromptTokens sums total_prompt_tokens across all records that share
// the most recent date. The activity API now returns one record per category,
// so a single record is just one slice of the day, not the full daily total.
func latestDayPromptTokens(records []ActivityRecord) (int64, bool) {
	if len(records) == 0 {
		return 0, false
	}
	maxDate := records[0].Date
	for i := 1; i < len(records); i++ {
		if records[i].Date > maxDate {
			maxDate = records[i].Date
		}
	}
	var total int64
	for i := range records {
		if records[i].Date == maxDate {
			total += records[i].TotalPromptTokens
		}
	}
	return total, true
}

// computePromptTokenDeltas returns per-model deltas of total_prompt_tokens
// since the previous snapshot, plus the new snapshot to use as next prev.
// On the first call (prev == nil) deltas is empty: we only seed the baseline.
// On UTC midnight rollover (latest < prev) the delta is the new running total.
func computePromptTokenDeltas(activity map[string][]ActivityRecord, prev map[string]int64) (map[string]int64, map[string]int64) {
	deltas := make(map[string]int64, len(activity))
	newPrev := make(map[string]int64, len(activity))

	for modelID, records := range activity {
		latest, ok := latestDayPromptTokens(records)
		if !ok {
			continue
		}
		newPrev[modelID] = latest

		if prev == nil {
			continue
		}
		p, ok := prev[modelID]
		if !ok {
			continue
		}
		d := latest - p
		if d < 0 {
			d = latest
		}
		deltas[modelID] = d
	}

	return deltas, newPrev
}
