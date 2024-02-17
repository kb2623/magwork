SOURCE_FILE := main

all: latex

luatex:
	lualatex $(SOURCE_FILE)
	biber $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)
	lualatex $(SOURCE_FILE)
