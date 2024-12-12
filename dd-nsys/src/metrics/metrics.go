package metrics

import (
	"context"
	"exporter/data"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/DataDog/datadog-api-client-go/v2/api/datadog"
	"github.com/DataDog/datadog-api-client-go/v2/api/datadogV2"
	"github.com/joho/godotenv"
)

func convertValue(unconverted_value reflect.Value) (float64, bool) {
	switch value := unconverted_value.Interface().(type) {
	case int:
		return float64(value), true
	case int64:
		return float64(value), true
	case float64:
		return value, true
	default:
		fmt.Println(reflect.TypeOf(unconverted_value.Interface()))
		return 0, false
	}
}
func SendMetricWithName(metric_prefix string, fields reflect.Type, values reflect.Value) {
	name := values.FieldByName("Name").String()
	for i := 0; i < fields.NumField(); i++ {
		field := fields.Field(i)
		if field.Name == "Name" {
			continue
		}
		metric_name := fmt.Sprintf("%s_%s_%s", metric_prefix, name, strings.ToLower(field.Name))
		metric_value, ok := convertValue(values.Field(i))
		if ok {
			submitMetrics(createPayload(metric_name, metric_value))
		} else {
			fmt.Println("Error converting metric:", metric_name)
		}
	}
}

func SendMetricWithKernelName(metric_prefix string, fields reflect.Type, values reflect.Value) {
	name_with_args := values.FieldByName("Name").String()
	kernel_name, kernel_args := data.SplitKernelName(name_with_args)
	for i := 0; i < fields.NumField(); i++ {
		field := fields.Field(i)
		if field.Name == "Name" {
			continue
		}
		metric_name := fmt.Sprintf("%s_%s_%s", metric_prefix, kernel_name, strings.ToLower(field.Name))
		metric_value, ok := convertValue(values.Field(i))
		if ok {
			submitMetrics(createKernelPayload(metric_name, metric_value, kernel_args))
		} else {
			fmt.Println("Error converting metric:", metric_name)
		}
	}
}

func SendMetricWithOperation(metric_prefix string, fields reflect.Type, values reflect.Value) {
	name := data.ConvertOperationName(values.FieldByName("Operation").String())
	for i := 0; i < fields.NumField(); i++ {
		field := fields.Field(i)
		if field.Name == "Operation" {
			continue
		}
		metric_name := fmt.Sprintf("%s_%s_%s", metric_prefix, name, strings.ToLower(field.Name))
		metric_value, ok := convertValue(values.Field(i))
		if ok {
			submitMetrics(createPayload(metric_name, metric_value))
		} else {
			fmt.Println("Error converting metric:", metric_name)
		}
	}
}

func createKernelPayload(metric_name string, value float64, kernel_args string) datadogV2.MetricPayload {
	return datadogV2.MetricPayload{
		Series: []datadogV2.MetricSeries{
			{
				Metric: metric_name,
				Type:   datadogV2.METRICINTAKETYPE_UNSPECIFIED.Ptr(),
				Points: []datadogV2.MetricPoint{
					{
						Timestamp: datadog.PtrInt64(time.Now().Unix()),
						Value:     datadog.PtrFloat64(value),
					},
				},
				Resources: []datadogV2.MetricResource{
					{
						Name: datadog.PtrString("gpu-tracing"),
						Type: datadog.PtrString("host"),
					},
					{
						Name: datadog.PtrString(kernel_args),
						Type: datadog.PtrString("kernel_args"),
					},
				},
			},
		},
	}
}

func createPayload(metric_name string, value float64) datadogV2.MetricPayload {
	return datadogV2.MetricPayload{
		Series: []datadogV2.MetricSeries{
			{
				Metric: metric_name,
				Type:   datadogV2.METRICINTAKETYPE_UNSPECIFIED.Ptr(),
				Points: []datadogV2.MetricPoint{
					{
						Timestamp: datadog.PtrInt64(time.Now().Unix()),
						Value:     datadog.PtrFloat64(value),
					},
				},
				Resources: []datadogV2.MetricResource{
					{
						Name: datadog.PtrString("gpu-tracing"),
						Type: datadog.PtrString("host"),
					},
				},
			},
		},
	}
}

func submitMetrics(body datadogV2.MetricPayload) {
	godotenv.Load()
	ctx := context.WithValue(
		context.Background(),
		datadog.ContextAPIKeys,
		map[string]datadog.APIKey{
			"apiKeyAuth": {
				Key: os.Getenv("DD_API_KEY"),
			},
		},
	)
	configuration := datadog.NewConfiguration()
	apiClient := datadog.NewAPIClient(configuration)
	api := datadogV2.NewMetricsApi(apiClient)
	_, _, _ = api.SubmitMetrics(ctx, body, *datadogV2.NewSubmitMetricsOptionalParameters())

	// if err != nil {
	// 	fmt.Fprintf(os.Stderr, "Error when calling `MetricsApi.SubmitMetrics`: %v\n", err)
	// 	fmt.Fprintf(os.Stderr, "Full HTTP response: %v\n", r)
	// }

	// responseContent, _ := json.MarshalIndent(resp, "", "  ")
	// fmt.Fprintf(os.Stdout, "Response from `MetricsApi.SubmitMetrics`:\n%s\n", responseContent)
}
