import pandas as pd
import librosa
import pickle
import os


queries_df = pd.read_pickle("./queries_with_midi.pkl")


def save_result_temp_folder(data, filename):
    # Serialize object to a file
    with open(f'./f0_queries/temp/{filename}.pickle', 'wb') as file:
        pickle.dump(data, file)

def get_result_from_temp_folder(filename):
    # Deserialize object from file
    file_path = f'./f0_queries/temp/{filename}.pickle'

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
    query_id = row['Query ID']
    result = get_f0(f"MTG-QBH/audio/{query_id}.wav", f"q_{query_id}")

    return pd.Series({'f0': result['f0'], 'voiced_flag': result['voiced_flag']})


def append_f0(df):
    query_ids = df["Query ID"].tolist()
    count = 0
    for query_id in query_ids:
        count = count+1
        print(count)
        print(f"Query Id: {query_id}")
        get_f0(f"MTG-QBH/audio/{query_id}.wav", f"q_{query_id}")
    
    print("Finish Caching Results ------ ")

    new_columns = df.apply(lambda row: process_row(row), axis=1)
    
    df = pd.concat([df, new_columns], axis=1)
    return df

queries_df = append_f0(queries_df)
queries_df.to_pickle("./queries_with_f0.pkl")

