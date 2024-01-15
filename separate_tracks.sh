# Directory containing audio files
inputDir="./songs_wav"
# Output directory for separated tracks
outputDir="./output"

# Loop through each audio file in the input directory
for file in "$inputDir"/*.wav; do
    # Extracting filename without extension
    filename=$(basename "$file" .wav)
    demucs "$file" -o "$outputDir"
done