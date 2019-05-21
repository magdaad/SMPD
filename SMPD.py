import tkinter
from tkinter import ttk
from tkinter import filedialog
import csv
import numpy
import math

filePath = ""


def loadFile():
    global filePath
    filePath = filedialog.askopenfilename()
    print(filePath)
    loadData(filePath)


def loadData(filePath):
    acerClass = []
    quercusClass = []
    with open(filePath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if "Acer" in row[0]:
                del row[0]
                acerClass.append(row)
            if "Quercus" in row[0]:
                del row[0]
                quercusClass.append(row)
    return acerClass, quercusClass


def calculateFeatures():
    if selected_method.get() == "Fisher":
        calculateFisher()
    if selected_method.get() == "Quercus":
        calculateSFS()


def calculateFisher():
    print("calculate fisher")
    # działa jak na razie tylko dla wyboru jednej cechy
    acer, quercus = loadData(filePath)
    acerClass = numpy.array(acer, dtype=float)
    quercusClass = numpy.array(quercus, dtype=float)
    acerMeans = numpy.mean(acerClass, axis=0)
    quercusMeans = numpy.mean(quercusClass, axis=0)

    maxFisher = -1.0
    bestIndex = -1

    for index in range(0, 64):
        # średnia dla cechy
        acerMean = acerMeans[index]
        quercusMean = quercusMeans[index]

        # wyciągnięte wartości dla jednej cechy ze wszystkich próbek - średnia
        acerValues = acerClass[:, index] - acerMean
        quercusValues = quercusClass[:, index] - quercusMean

        # odchylenie standardowe dla acer i quercus
        acerDeviation = math.sqrt((sum([i ** 2 for i in acerValues]))/acerClass.shape[0])
        quercusDeviation = math.sqrt((sum([i ** 2 for i in quercusValues]))/quercusClass.shape[0])

        # fisher dla danej cechy
        fisher = abs(acerMean - quercusMean)/(acerDeviation + quercusDeviation)
        if fisher > maxFisher:
            maxFisher=fisher
            bestIndex=index

    print(maxFisher)
    print(bestIndex)
    listbox.insert(tkinter.END, "index najlepszej cechy: " + str(bestIndex) + " wartość fisher: " + str(maxFisher))
    # printuje index a nie numer cechy (index+1)



def calculateSFS():
    print("calculate sfs")


main = tkinter.Tk()
main.title('SMPD')
main.minsize(800, 500)
selected_method = tkinter.StringVar("")
# Defines and places the notebook widget
notebook = ttk.Notebook(main)
notebook.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')

# Adds tab 1 of the notebook
page1 = ttk.Frame(notebook)
notebook.add(page1, text='Festure selection')
page1.rowconfigure(0, weight=1)
page1.rowconfigure(1, weight=1)
page1.rowconfigure(2, weight=100)

# load file button
load_button = ttk.Button(page1, text="Load file", cursor="hand2", command=lambda: loadFile())
load_button.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

# choose how many features
combobox = ttk.Combobox(page1, state="readonly",
                        values=[i for i in range(1,65)])
combobox.grid(row=0, column=1, sticky="n", padx=20, pady=20)

# choose if fisher or sfs
fisher_radiobutton = ttk.Radiobutton(page1, text="Fisher",  value="Fisher",
                                                variable=selected_method, cursor="hand2")
fisher_radiobutton.grid(row=1, column=1, sticky="nw", padx=20)
sfs_radiobutton = ttk.Radiobutton(page1, text="SFS",  value="SFS", variable=selected_method,
                                                cursor="hand2")
sfs_radiobutton.grid(row=2, column=1, sticky="nw", padx=20)

# show results in listbox
listbox = tkinter.Listbox(page1, activestyle="none",
                    height=30, width=70)
listbox.grid(row=0, column=3, rowspan=3, padx=20, pady=20)

# calculate button
calculate_button = ttk.Button(page1, text="Calculate", cursor="hand2",command=lambda: calculateFeatures())
calculate_button.grid(row=0, column=2, padx=20, pady=20, sticky="nw")

# Adds tab 2 of the notebook
page2 = ttk.Frame(notebook)
notebook.add(page2, text='Classifiers')

main.mainloop()
