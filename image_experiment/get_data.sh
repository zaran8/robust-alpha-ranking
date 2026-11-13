#!/bin/bash
# Download the raw files of the five data sets of Rezende et al. (2014), about 1.2 GB,
# into DATA/raw.  Run it ONCE on the Spartan LOGIN node (compute nodes may have no internet):
#     bash get_data.sh                                        # into ./data/raw
#     bash get_data.sh /data/gpfs/projects/punim3024/alpha_data   # or another folder
# Files already present that pass their check are skipped, so it is safe to run it again.
# MNIST, CIFAR-10 and SVHN are checked against their published MD5 sums; Frey faces and
# small NORB have none, so their format and size are checked when they are first read.
set -uo pipefail
DATA=${1:-${DATA:-data}}
mkdir -p "$DATA/raw" || exit 1
cd "$DATA/raw" || exit 1

download () {   # download URL OUTPUT
  if command -v wget >/dev/null 2>&1; then
    wget -nv --tries=3 --timeout=120 -O "$2" "$1"
  else
    curl -fL --retry 3 --connect-timeout 60 -o "$2" "$1"
  fi
}

good () {       # good FILE MD5|-
  [ -s "$1" ] || return 1
  if [ "$2" != "-" ]; then echo "$2  $1" | md5sum -c --status - || return 1; fi
  case "$1" in *.gz) gzip -t "$1" 2>/dev/null || return 1 ;; esac
  return 0
}

FAILED=()
fetch () {      # fetch FILE MD5|- URL [URL ...]
  local f=$1 md5=$2; shift 2
  if good "$f" "$md5"; then echo "ok      $f (already here)"; return 0; fi
  rm -f "$f"
  for url in "$@"; do
    echo "get     $f  <-  $url"
    if download "$url" "$f.part" && good "$f.part" "$md5"; then
      mv "$f.part" "$f"; echo "ok      $f"; return 0
    fi
    echo "        failed or bad checksum from $url"
    rm -f "$f.part"
  done
  FAILED+=("$f")
}

# MNIST (MD5 sums as in torchvision)
for f in train-images-idx3-ubyte.gz:f68b3c2dcbeaaa9fbdd348bbdeb94873 \
         t10k-images-idx3-ubyte.gz:9fb629c4189551a2d022fa330f9573f3; do
  n=${f%%:*}; m=${f##*:}
  fetch "$n" "$m" "https://ossci-datasets.s3.amazonaws.com/mnist/$n" \
                  "https://storage.googleapis.com/cvdf-datasets/mnist/$n" \
                  "http://yann.lecun.com/exdb/mnist/$n"
done
# Frey faces (Sam Roweis' data page)
fetch frey_rawface.mat - "https://cs.nyu.edu/~roweis/data/frey_rawface.mat"
# small NORB (images only: the -dat files)
for n in smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz \
         smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz; do
  fetch "$n" - "https://cs.nyu.edu/~yann/data/norb-v1.0-small/$n" \
               "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/$n"
done
# CIFAR-10, binary version (no pickle needed); MD5 from the CIFAR page
fetch cifar-10-binary.tar.gz c32a1d4ab5d03f1284b67883e8d87530 \
      "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" \
      "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
# SVHN, cropped digits (MD5 sums as in torchvision)
fetch train_32x32.mat e26dedcc434d2e4c54c9b2d4a06d8373 "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
fetch test_32x32.mat eb5a983be6a315427106f1b164d9cef3 "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

echo
ls -l
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  echo "Download these by hand into $(pwd) and run this script again."
  exit 1
fi
echo "All files downloaded into $(pwd)."
echo "Next:  bash submit_images.sh"
