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

	// Provider-level chart series: provider slug -> [points]. Each point has
	// a UTC date and a map of raw-model-id -> running token count for that day.
	ProviderActivity map[string][]ProviderActivityPoint
	// ProviderTokenDeltas[provider][modelID] = tokens added since last scrape
	// to today's running bucket. Empty on first scrape (baseline).
	ProviderTokenDeltas map[string]map[string]int64
	ProviderActivityErrors int
}

// Re-export client types for cache consumers
type Model = client.Model
type EndpointsResponse = client.EndpointsResponse
type Endpoint = client.Endpoint
type ActivityRecord = client.ActivityRecord
type ProviderActivityPoint = client.ProviderActivityPoint

type Cache struct {
	mu                    sync.RWMutex
	data                  *CachedData
	client                *client.OpenRouterClient
	interval              time.Duration
	activityInterval      time.Duration
	stopCh                chan struct{}
	logger                *slog.Logger
	activityModels        []string
	activityProviders     []string
	sessionCookie         string
	prevPromptTokens      map[string]int64
	// prevTodayTokens[provider][modelID] = today's running total seen on the
	// previous scrape, used to compute intraday delta. The "today" bucket is
	// the latest x in the chart; it grows during the day, resets at UTC
	// midnight when a new bucket appears.
	prevTodayTokens       map[string]map[string]int64
	prevTodayDate         map[string]string // provider -> date of the bucket prevTodayTokens belongs to
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

func (c *Cache) SetProviderActivity(providers []string) {
	c.activityProviders = providers
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

	// Fetch provider chart data if configured
	var provDeltas map[string]map[string]int64
	var newProvPrev map[string]map[string]int64
	var newProvDate map[string]string
	if len(c.activityProviders) > 0 && c.sessionCookie != "" {
		provResult, err := c.client.FetchAllProviderActivity(ctx, c.activityProviders, c.sessionCookie)
		if err != nil {
			c.logger.Error("fetch provider activity failed", "error", err)
		} else {
			cached.ProviderActivity = provResult.Activity
			cached.ProviderActivityErrors = provResult.Errors
			provDeltas, newProvPrev, newProvDate = computeProviderTokenDeltas(provResult.Activity, c.prevTodayTokens, c.prevTodayDate)
			cached.ProviderTokenDeltas = provDeltas
		}
	}

	c.mu.Lock()
	c.data = cached
	if newPrev != nil {
		c.prevPromptTokens = newPrev
	}
	if newProvPrev != nil {
		c.prevTodayTokens = newProvPrev
		c.prevTodayDate = newProvDate
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

	// Provider chart fetch (independent — failure here doesn't kill the per-model path)
	var provActivity map[string][]ProviderActivityPoint
	var provErrors int
	var provDeltas map[string]map[string]int64
	var newProvPrev map[string]map[string]int64
	var newProvDate map[string]string
	if len(c.activityProviders) > 0 {
		provResult, perr := c.client.FetchAllProviderActivity(ctx, c.activityProviders, c.sessionCookie)
		if perr != nil {
			c.logger.Error("provider activity refresh failed", "error", perr)
		} else {
			provActivity = provResult.Activity
			provErrors = provResult.Errors
			provDeltas, newProvPrev, newProvDate = computeProviderTokenDeltas(provResult.Activity, c.prevTodayTokens, c.prevTodayDate)
		}
	}

	c.mu.Lock()
	if c.data != nil {
		c.data.Activity = activityResult.Activity
		c.data.ActivityErrors = activityResult.Errors
		c.data.PromptTokenDeltas = deltas
		if provActivity != nil {
			c.data.ProviderActivity = provActivity
			c.data.ProviderActivityErrors = provErrors
			c.data.ProviderTokenDeltas = provDeltas
		}
	}
	c.prevPromptTokens = newPrev
	if newProvPrev != nil {
		c.prevTodayTokens = newProvPrev
		c.prevTodayDate = newProvDate
	}
	c.mu.Unlock()

	c.logger.Info("activity refreshed",
		"models", len(activityResult.Activity),
		"errors", activityResult.Errors,
		"providers", len(provActivity),
		"provider_errors", provErrors,
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

// latestPoint returns the chart point with the largest x value, which is
// today's running bucket. Returns nil if the series is empty.
func latestPoint(series []ProviderActivityPoint) *ProviderActivityPoint {
	if len(series) == 0 {
		return nil
	}
	idx := 0
	for i := 1; i < len(series); i++ {
		if series[i].X > series[idx].X {
			idx = i
		}
	}
	return &series[idx]
}

// computeProviderTokenDeltas computes intraday deltas of today's running token
// counts per (provider, model) since the previous scrape. Returns empty deltas
// on the first scrape (only seeds prev), and zero (not negative) on UTC
// midnight rollover when a new "today" bucket appears.
func computeProviderTokenDeltas(
	activity map[string][]ProviderActivityPoint,
	prev map[string]map[string]int64,
	prevDate map[string]string,
) (map[string]map[string]int64, map[string]map[string]int64, map[string]string) {
	deltas := make(map[string]map[string]int64, len(activity))
	newPrev := make(map[string]map[string]int64, len(activity))
	newDate := make(map[string]string, len(activity))

	for provider, series := range activity {
		latest := latestPoint(series)
		if latest == nil {
			continue
		}
		todayDate := latest.X
		if len(todayDate) >= 10 {
			todayDate = todayDate[:10]
		}

		// Snapshot the latest ys so the next scrape can diff against it.
		snap := make(map[string]int64, len(latest.Ys))
		for k, v := range latest.Ys {
			snap[k] = v
		}
		newPrev[provider] = snap
		newDate[provider] = todayDate

		// First scrape, or rollover to a new day: emit no deltas (baseline only).
		if prev == nil {
			continue
		}
		pPrev, ok := prev[provider]
		if !ok {
			continue
		}
		if prevDate != nil && prevDate[provider] != todayDate {
			// New day — yesterday's bucket finalized. Skip emitting deltas
			// this tick to avoid an artificial spike.
			continue
		}

		modelDeltas := make(map[string]int64, len(latest.Ys))
		for modelID, cur := range latest.Ys {
			p, had := pPrev[modelID]
			if !had {
				continue // model just appeared
			}
			d := cur - p
			if d < 0 {
				d = 0 // late-arriving correction; clamp
			}
			modelDeltas[modelID] = d
		}
		if len(modelDeltas) > 0 {
			deltas[provider] = modelDeltas
		}
	}

	return deltas, newPrev, newDate
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
