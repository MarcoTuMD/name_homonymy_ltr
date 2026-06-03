from scipy.stats import wilcoxon

recall_1_model_a = [0.8337, 0.8369, 0.8394, 0.8290, 0.8254]
recall_1_model_b = [0.8497, 0.8468, 0.8434, 0.8270, 0.8214]

recall_5_model_a = [0.9880, 0.9882, 0.9859, 0.9841, 0.9841]
recall_5_model_b = [0.9840, 0.9902, 0.9799, 0.9861, 0.9901]

recall_10_model_a = [0.9980, 1.0000, 0.9960, 0.9980, 1.0000]
recall_10_model_b = [0.9980, 1.0000, 0.9960, 0.9980, 1.0000]

recall_15_model_a = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
recall_15_model_b = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

recall_20_model_a = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
recall_20_model_b = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]


mrr_a = [0.9050, 0.9089, 0.9059, 0.9013, 0.8984]
mrr_b = [0.9131, 0.9125, 0.9084, 0.9018, 0.8956]

for k in [1, 5, 10, 15, 20]:
    stat, p_value = wilcoxon(
        eval(f"recall_{k}_model_b"),
        eval(f"recall_{k}_model_a"),
        alternative='greater'
    )
    print(f"Recall@{k} - Wilcoxon statistic: {stat}, p-value: {p_value}")

stat, p_value = wilcoxon(mrr_b, mrr_a, alternative='greater')
print(f"MRR - Wilcoxon statistic: {stat}, p-value: {p_value}")

# stat, p_value = wilcoxon(recall_10_model_b, recall_10_model_a, alternative='greater')

# print("Wilcoxon statistic:", stat)
# print("p-value:", p_value)