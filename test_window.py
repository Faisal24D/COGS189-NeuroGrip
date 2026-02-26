from psychopy import visual, event, core

win = visual.Window(size=(900, 600), color="black")
text = visual.TextStim(win, text="Window test: press SPACE", color="white")

while True:
    text.draw()
    win.flip()
    keys = event.getKeys()
    if "space" in keys:
        break

win.close()
core.quit()