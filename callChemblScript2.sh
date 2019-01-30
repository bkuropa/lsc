#!/bin/bash

#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

echo "Calling chembl script"

rawDataDir=fingerprints
sdfFile=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

mkdir $DIR/scripts/raw/$rawDataDir/SHED
$DIR/lsc/chemblScript2.sh SHED ELEMENT_SYMBOL $DIR/scripts/sdfs/$sdfFile $DIR/scripts/raw/$rawDataDir/SHED chembl_id STRING_PATTERNS

mkdir $DIR/scripts/raw/$rawDataDir/CATS2D
$DIR/lsc/chemblScript2.sh CATS2D ELEMENT_SYMBOL $DIR/scripts/sdfs/$sdfFile $DIR/scripts/raw/$rawDataDir/CATS2D chembl_id STRING_PATTERNS





dirName=$DIR/scripts/raw/$rawDataDir/SHED
outFile=$2

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done



dirName=$DIR/scripts/raw/$rawDataDir/CATS2D
outFile=$3

rm $dirName/../$outFile
for i in `ls $dirName/myout*.res`; do
  echo $i
  head -n`wc -l $i |cut -f 1 -d " "` $i >> $dirName/../$outFile
done
