SOURCE_FILE := main
old_main := main_$(shell date '+%Y-%m-%d-%H-%M-%S').pdf

all: luatex

luatex:
	lualatex $(SOURCE_FILE)
	biber $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)

build: luatex

backup:
	cp main.pdf $(old_main)

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.loa *.log *.lot *.lof *.out *.toc *.idx *.run.xml
ifneq (,$(wildcard ./main.pdf))
	make backup
endif
	rm -f main.pdf

clean_all: clean
	rm -f *.pdf
