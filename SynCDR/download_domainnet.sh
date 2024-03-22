#!/bin/bash

curr_dir=$(pwd)

mkdir -p $1
cd $1
mkdir domainnet
cd domainnet

for domain in clipart painting sketch;
do
    echo "Downloading $domain"
    wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/${domain}.zip -o ${domain}.zip
    echo "Unzipping $domain"
    unzip -q ${domain}.zip
    rm ${domain}.zip
done

cd $curr_dir

