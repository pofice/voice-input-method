from funasr import AutoModel

model = AutoModel(model="paraformer", device="cpu")

res = model.export(quantize=True)