# Search root and destination
SRC_ROOT="/media/storage/vivib"
DEST="/home/vivib/emoca/emoca/dataset/converted_mp4"
mkdir -p "$DEST"

# Convert ONLY videos inside folders whose name contains "MD"
# and whose filename ends with .avi1.avi or .avi1
find "$SRC_ROOT" -type f -regextype posix-extended \
  -iregex '.*/[^/]*[[:space:]]*MD[^/]*/[^/]+\.avi1(\.avi)?$' -print0 |
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  # keep original video name, just change extension to .mp4
  if [[ "$base" =~ \.avi1\.avi$ ]]; then
    stem="${base%.avi1.avi}"
  elif [[ "$base" =~ \.avi1$ ]]; then
    stem="${base%.avi1}"
  else
    continue
  fi
  out="$DEST/${stem}.mp4"

  # skip if already converted
  if [[ -s "$out" ]]; then
    echo "SKIP (exists): $out"
    continue
  fi

  echo "→ Converting: $f"
  ffmpeg -nostdin -hide_banner -loglevel warning -stats -y \
    -analyzeduration 10M -probesize 10M -fflags +genpts -err_detect ignore_err \
    -i "$f" -map 0:v:0 -map 0:a:0? \
    -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -vsync 1 \
    -c:a aac -b:a 128k -movflags +faststart \
    "$out"
done

echo "All done. Outputs in: $DEST"
