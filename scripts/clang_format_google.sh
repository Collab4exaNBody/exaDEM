#!/usr/bin/env bash

# Chemin vers clang-format
CLANG_FORMAT="$HOME/.local/bin/clang-format"

# Style Google avec limite de colonne à 160
STYLE='{BasedOnStyle: Google, ColumnLimit: 120, , SortIncludes: false}'

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <file1.cpp> [file2.cpp ...]"
  exit 1
fi

for file in "$@"; do
  if [ ! -f "$file" ]; then
    echo "Erreur : fichier '$file' introuvable"
    continue
  fi

  history_file="${file}.history"
  tmp_file="$(mktemp)"

  # Sauvegarde
  cp "$file" "$history_file"

  # Formatage
  "$CLANG_FORMAT" -style="$STYLE" "$file" > "$tmp_file"

  # Remplacement atomique
  mv "$tmp_file" "$file"

  echo "✔ Formaté : $file (backup -> $history_file)"
done
