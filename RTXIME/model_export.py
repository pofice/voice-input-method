from funasr import AutoModel

model = AutoModel(model="paraformer-zh", device="cpu")

res = model.export(quantize=True)