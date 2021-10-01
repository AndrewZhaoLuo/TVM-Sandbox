import octomizer
import octomizer.client

client = octomizer.client.OctomizerClient()

models = client.list_models()

# T4 AWS + GCP targets
platforms = ["aws_g4dn.xlarge", "nvidia_tesla_t4"]

for model in models:
    model_variants = list(model.list_model_variants())
    print(f"Model: {model.uuid}")
    for variant in model_variants:
        print(f"\tVariant: {variant.uuid}")
        for platform in platforms:
            print(f"\t\tBenchmarking on platform {platform}")
            workflow = variant.benchmark(platform=platform, num_benchmark_trials=10, num_runs_per_trial=100)
