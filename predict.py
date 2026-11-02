import pickle

with open("model.bin", "rb") as f_out:
    pipeline = pickle.load(f_out)

sample = {'category': 'Education', 
 'duration_sec': 1201, 
 'title_length': 72, 
 'thumbnail_face': 0, 
 'upload_hour': 13, 
 'upload_day': 'Sat', 
 'tags_count': 15, 
 'description_length': 569, 
 'subscriber_count': 2789543}

sample_pred = pipeline.predict_proba([sample])[:, 1]

print(sample_pred)