SOURCE_FILE := main

all: luatex

luatex:
	lualatex $(SOURCE_FILE)
	biber $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)
