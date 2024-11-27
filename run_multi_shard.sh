# Directory containing the .arrow files
arrow_dir="/home/ajay.meena/obelics/huggiface_cache/HuggingFaceM4___obelics/default/0.0.0/08b7dca61ec09bacd68234ba6c388209b126d31c"

# Python script to process the files
python_script="with_async_objectstorage.py"

find "$arrow_dir" -name "obelics-train-*.arrow" \
    | grep -E "obelics-train-00(1[0-9][0-9]|)-of-01439\.arrow" \
    | xargs -P 5 -I {} python "$python_script" --arrow_file {}