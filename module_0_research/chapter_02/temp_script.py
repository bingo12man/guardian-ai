import json

with open("results/e6_results.json", encoding="utf-8") as f:
    data = json.load(f)

for record in data["evaluation"]["records"]:
    task = record["task_id"]

    if task not in [
        "missing_profit",
        "customers_difference",
        "energy_difference"
    ]:
        continue

    print("\n" + "=" * 55)
    print("TASK:", task)
    print("CONDITION:", record["condition"])
    print("FILES:", record["outcome"]["files"])
    print("ANSWER:", record["outcome"]["answer"])
    print("EVALUATION:", record["evaluation"])