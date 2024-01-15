import pandas as pd
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

import pickle
import os


songs_df = pd.read_csv("./MTG-QBH/metadata/Collection_Canonicals.csv", encoding='latin1')


def save_result_temp_folder(data, filename):
    # Serialize object to a file
    with open(f'./temp/{filename}.pickle', 'wb') as file:
        pickle.dump(data, file)

def get_result_from_temp_folder(filename):
    # Deserialize object from file
    file_path = f'./temp/{filename}.pickle'

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            loaded_object = pickle.load(file)
            return loaded_object
    return None
 
count = 0 

def printCount():
    global count
    print(count)
    count += 1
# FUNCTIONS
def get_song_vocals_midi(song_id):
    printCount()
    cached_result = get_result_from_temp_folder(str(song_id))
    if(cached_result is not None):
        print(f"Cached!")
        return  cached_result
    model_output, vocal_midi_data, note_events = predict(f"output/htdemucs/{song_id}/vocals.wav")
    del model_output
    del note_events
    save_result_temp_folder(vocal_midi_data, str(song_id))
    return vocal_midi_data

def append_vocals_midi(df):
    song_ids = df["Song ID"].tolist()
    for song_id in song_ids:
        get_song_vocals_midi(str(song_id))
    
    print("Finish Caching Results ------ ")
    df["vocals_midi"] = df["Song ID"].apply(lambda x: get_song_vocals_midi(str(x)))
    return df

songs_df = append_vocals_midi(songs_df)
songs_df.to_pickle("./songs_with_midi.pkl")