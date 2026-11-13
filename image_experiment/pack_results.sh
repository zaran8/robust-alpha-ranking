#!/bin/bash
# Collect the results to send back (a few MB):  bash pack_results.sh
shopt -s nullglob
FILES=(results_images/*.csv results_images/*.tex results_images/*.png
       results_images/cache/*/settings.json results_images/cache/*/chosen_*.json
       results_images/cache/*/tuning_*.csv results_images/cache/*/grid_extensions_*.json
       "${DATA:-data}"/prepared/*.json logs/alpha_img*.out)
if [ ${#FILES[@]} -eq 0 ]; then echo "nothing to pack yet"; exit 1; fi
tar czf results_images_pack.tar.gz "${FILES[@]}"
echo "wrote $(pwd)/results_images_pack.tar.gz ($(du -h results_images_pack.tar.gz | cut -f1))"
