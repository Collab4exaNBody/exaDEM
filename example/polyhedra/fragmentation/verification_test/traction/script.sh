#!/bin/bash

# Définition de la liste des valeurs
NAMES=("5e-3" "1e-2" "2e-2" "4e-2" "8e-2" "1e-1")

# Définition de la fonction
process_traction() {
    local val=$1
    
    # Copie du fichier template
    cp traction.msp "traction_${val}.msp"
    
    # Remplacement de la chaîne GC par la valeur actuelle
    # On utilise "" autour de ${val} pour que sed interprète bien la variable
    sed -i "s/GC/${val}/g" "traction_${val}.msp"
    
    # Extraction des données avec awk
    # $2-1.5 : on soustrait 1.5 à la 2ème colonne
    # $8 * (-1) : on inverse le signe de la 8ème colonne
    echo "../exaDEM traction_${val}.msp --omp_num_threads 1"
}

cmd_awk() {
    local val=$1
		T2='$2'
		T8='$8'
		echo "awk 'FNR == 4 {print (${T2}-1.5)/0.98 \" \" ${T8} * (-1) / (0.98) }' "XYZFiles${val}"/*.xyz > "res_${val}.txt""
}

echo "Run script"

# Boucle sur chaque élément du tableau
for val in "${NAMES[@]}"
do
    process_traction "$val"
done
for val in "${NAMES[@]}"
do
    cmd_awk "$val"
done
