import re
with open('shap_7day_analysis_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()
new_func = '''def compute_shap_importance(model, dataset, window_size, device, bg_samples=100, exp_samples=500):
    bg_samples = min(bg_samples, len(dataset))
    exp_samples = min(exp_samples, len(dataset))
    
    background_indices = np.random.choice(len(dataset), bg_samples, replace=False)
    background = torch.stack([dataset[i][0] for i in background_indices]).numpy()
    background = background.reshape(background.shape[0], -1)

    sample_indices = np.random.choice(len(dataset), exp_samples, replace=False)
    samples = torch.stack([dataset[i][0] for i in sample_indices]).numpy()
    samples_flat = samples.reshape(samples.shape[0], -1)

    def predict_fn(x_flat):
        x = x_flat.reshape(-1, window_size, 7)
        x_tensor = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()
        return pred

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(samples_flat, nsamples=100)

    shap_arr = np.array(shap_values)  # (4, n_samples, window_size*7)
    shap_arr = shap_arr.reshape(4, -1, window_size, 7)
    importance = np.mean(np.abs(shap_arr), axis=(0, 1, 3))  # (window_size,)
    return importance, explainer, samples_flat, shap_values
'''
# 替换函数（从 def compute_shap_importance 到该函数结束）
pattern = r'def compute_shap_importance\(.*?\):(?:.*?\n)*?    return importance, explainer, samples_flat, shap_values'
content = re.sub(pattern, new_func, content, flags=re.DOTALL)
with open('shap_7day_analysis_complete.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("修复完成")
