
test:
	python3 -m unittest discover

clean:
	-$(RM) models/*.csv
	-$(RM) models/*.pth
	-$(RM) models/*.png
