package collector

import "github.com/prometheus/client_golang/prometheus"

var (
	MetricInputPrice = prometheus.NewDesc(
		"openrouter_model_input_price_dollars_per_million_tokens",
		"Price per million input tokens in USD",
		[]string{"model_id", "provider_name", "tag", "quantization"}, nil,
	)
	MetricOutputPrice = prometheus.NewDesc(
		"openrouter_model_output_price_dollars_per_million_tokens",
		"Price per million output tokens in USD",
		[]string{"model_id", "provider_name", "tag", "quantization"}, nil,
	)
	MetricCacheReadPrice = prometheus.NewDesc(
		"openrouter_model_input_cache_read_price_dollars_per_million_tokens",
		"Price per million cached input tokens in USD",
		[]string{"model_id", "provider_name", "tag", "quantization"}, nil,
	)

	MetricUptime30m = prometheus.NewDesc(
		"openrouter_endpoint_uptime_percentage_last_30m",
		"Endpoint uptime percentage over the last 30 minutes",
		[]string{"model_id", "provider_name", "tag", "quantization"}, nil,
	)
	MetricUptime5m = prometheus.NewDesc(
		"openrouter_endpoint_uptime_percentage_last_5m",
		"Endpoint uptime percentage over the last 5 minutes",
		[]string{"model_id", "provider_name", "tag", "quantization"}, nil,
	)
	MetricUptime1d = prometheus.NewDesc(
		"openrouter_endpoint_uptime_percentage_last_1d",
		"Endpoint uptime percentage over the last day",
		[]string{"model_id", "provider_name", "tag", "quantization"}, nil,
	)

	MetricThroughput = prometheus.NewDesc(
		"openrouter_endpoint_throughput_tokens_per_second",
		"Throughput in tokens per second (requires API key)",
		[]string{"model_id", "provider_name", "tag", "quantization", "quantile"}, nil,
	)
	MetricLatency = prometheus.NewDesc(
		"openrouter_endpoint_latency_milliseconds",
		"Latency in milliseconds (requires API key)",
		[]string{"model_id", "provider_name", "tag", "quantization", "quantile"}, nil,
	)

	MetricModelInfo = prometheus.NewDesc(
		"openrouter_model_info",
		"Static model metadata as labels; value is always 1",
		[]string{"model_id", "model_name", "context_length", "modality", "tokenizer", "input_modalities", "output_modalities"}, nil,
	)

	MetricScrapeDuration = prometheus.NewDesc(
		"openrouter_scrape_duration_seconds",
		"Duration of the last cache refresh in seconds",
		nil, nil,
	)
	MetricScrapeErrors = prometheus.NewDesc(
		"openrouter_scrape_errors_total",
		"Total number of endpoint fetch errors in the last cache refresh",
		nil, nil,
	)
	MetricModelsScraped = prometheus.NewDesc(
		"openrouter_models_scraped",
		"Number of models in the last cache refresh",
		nil, nil,
	)
	MetricEndpointsScraped = prometheus.NewDesc(
		"openrouter_endpoints_scraped",
		"Number of endpoints in the last cache refresh",
		nil, nil,
	)
	MetricScrapeTimestamp = prometheus.NewDesc(
		"openrouter_scrape_timestamp_seconds",
		"Unix timestamp of the last successful cache refresh",
		nil, nil,
	)

	// Activity metrics
	MetricActivityRequests = prometheus.NewDesc(
		"openrouter_model_activity_requests",
		"Total requests per day for a model",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityPromptTokens = prometheus.NewDesc(
		"openrouter_model_activity_prompt_tokens",
		"Total prompt (input) tokens per day for a model",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityOutputTokens = prometheus.NewDesc(
		"openrouter_model_activity_output_tokens",
		"Total output tokens per day for a model (completion + reasoning)",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityCompletionTokens = prometheus.NewDesc(
		"openrouter_model_activity_completion_tokens",
		"Completion tokens per day (output minus reasoning)",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityCacheHitTokens = prometheus.NewDesc(
		"openrouter_model_activity_cache_hit_tokens",
		"Total cached tokens per day for a model",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityReasoningTokens = prometheus.NewDesc(
		"openrouter_model_activity_reasoning_tokens",
		"Total reasoning tokens per day for a model",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityInputOutputRatio = prometheus.NewDesc(
		"openrouter_model_activity_input_output_ratio",
		"Ratio of prompt tokens to output tokens per day",
		[]string{"model_id", "date"}, nil,
	)
	MetricActivityEstCostDollars = prometheus.NewDesc(
		"openrouter_model_activity_est_cost_dollars",
		"Estimated cost in USD based on the first endpoint pricing for the model",
		[]string{"model_id", "date"}, nil,
	)

	MetricActivityScrapeDuration = prometheus.NewDesc(
		"openrouter_activity_scrape_duration_seconds",
		"Duration of the last activity scrape in seconds",
		nil, nil,
	)
	MetricActivityScrapeErrors = prometheus.NewDesc(
		"openrouter_activity_scrape_errors_total",
		"Total number of activity fetch errors in the last refresh",
		nil, nil,
	)
	MetricActivityScrapeTimestamp = prometheus.NewDesc(
		"openrouter_activity_scrape_timestamp_seconds",
		"Unix timestamp of the last successful activity refresh",
		nil, nil,
	)

	MetricActivityPromptTokens5mDelta = prometheus.NewDesc(
	    "openrouter_model_activity_prompt_tokens_delta",
	    "Approximate prompt tokens added in the last 5-minute scrape interval (delta of the daily running total; resets at UTC midnight)",
	    []string{"model_id"}, nil,
	)

	MetricProviderTokensDaily = prometheus.NewDesc(
		"openrouter_provider_tokens_daily",
		"Tokens processed by a provider for a given model on a given UTC day. Today's value is the running total so far and grows intraday.",
		[]string{"provider", "model_id", "date"}, nil,
	)
	MetricProviderTokensIntradayDelta = prometheus.NewDesc(
		"openrouter_provider_tokens_intraday_delta",
		"Tokens added to today's running bucket since the previous activity scrape, per (provider, model). Empty on the first scrape and across UTC midnight.",
		[]string{"provider", "model_id"}, nil,
	)
	MetricProviderActivityScrapeErrors = prometheus.NewDesc(
		"openrouter_provider_activity_scrape_errors_total",
		"Number of provider-page fetch errors in the last refresh.",
		nil, nil,
	)
)
