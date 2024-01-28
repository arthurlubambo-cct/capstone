import pandas as pd
import librosa
import pickle
import os


songs_df = pd.read_pickle("./songs_with_midi.pkl")


def save_result_temp_folder(data, filename):
    # Serialize object to a file
    with open(f'./f0/temp/{filename}.pickle', 'wb') as file:
        pickle.dump(data, file)

def get_result_from_temp_folder(filename):
    # Deserialize object from file
    file_path = f'./f0/temp/{filename}.pickle'

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
def get_f0(audio_path, identifier):
    cached_result = get_result_from_temp_folder(str(identifier))
    if(cached_result is not None):
        print(f"{identifier} Cached!")
        return  cached_result
    obj = get_f0_from_audio_path(audio_path)
    save_result_temp_folder(obj, str(identifier))
    return obj

def get_f0_from_audio_path(audio_path):
    y, sr = librosa.load(audio_path)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
    obj = {}
    obj['f0'] = f0
    obj['voiced_flag'] = voiced_flag
    return obj

def process_row(row):
    song_id = row['Song ID']
    # Perform your logic here to generate values for the new columns
    result_vocals = get_f0(f"output/htdemucs/{song_id}/vocals.wav", f"vocals_{song_id}")
    result_other = get_f0(f"output/htdemucs/{song_id}/other.wav",f"other_{song_id}")

    return pd.Series({'vocals_f0': result_vocals['f0'], 'vocals_f0_voiced_flag': result_vocals['voiced_flag'],
                      'other_f0': result_other['f0'], 'other_f0_voiced_flag': result_other['voiced_flag']})


def append_f0(df):
    song_ids = df["Song ID"].tolist()
    count = 0
    for song_id in song_ids:
        count = count+1
        print(count)
        print(f"Song Id: {song_id}")
        get_f0(f"output/htdemucs/{song_id}/vocals.wav", f"vocals_{song_id}")
        get_f0(f"output/htdemucs/{song_id}/other.wav",f"other_{song_id}")
    
    print("Finish Caching Results ------ ")

    new_columns = df.apply(lambda row: process_row(row), axis=1)
    
    df = pd.concat([df, new_columns], axis=1)
    return df

songs_df = append_f0(songs_df)
songs_df.to_pickle("./songs_with_f0.pkl")

