#!/bin/bash
for f in $(find ./translations -name "*.md");
do
  echo Processing ${f}...
  pandoc ${f} --pdf-engine=xelatex -V header-includes:'\setromanfont{Jomolhari:style=Regular}' -o ${f%.md}.pdf
done

echo Uploading to S3...
aws s3 cp --exclude="*" --include="*.pdf" --recursive . s3://compassionai/public/

echo Cleaning up...
rm $(find ./translations -name "*.pdf")

echo Done!
