SOURCE_FILE := main

all: luatex

luatex:
	lualatex $(SOURCE_FILE)
	biber $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.loa *.log *.lot *.lof *.out *.toc *.idx *.run.xml

clean_all: clean
	rm -f *.pdf