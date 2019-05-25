import tkinter
from tkinter import ttk
from tkinter import filedialog
import csv
import numpy
import math
import itertools
import random
import sys

filePath = ""
trainSet = []
testSet = []
totalSet = []

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
    if selected_method.get() == "SFS":
        calculateSFS()

def calculateFisher():
    print("calculate fisher")
    acer, quercus = loadData(filePath)
    acerClass = numpy.array(acer, dtype=float)
    quercusClass = numpy.array(quercus, dtype=float)
    acerMeans = numpy.mean(acerClass, axis=0)
    quercusMeans = numpy.mean(quercusClass, axis=0)

    maxFisher = -1.0
    bestIndex = -1
    bestIndexes = ()

    # dodajemy 1 bo zbiera index a nie ilość cech
    selectedNumberOfFeatures = combobox.current() + 1
    if(selectedNumberOfFeatures == 1):
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
                maxFisher = fisher
                bestIndex = index

        listbox.insert(tkinter.END, "index najlepszej cechy: " + str(bestIndex) + " wartość fisher: " + str(maxFisher))
        # printuje index a nie numer cechy (index+1)
    else:
        print("multiple")
        # dla wszystkich możliwych kombinacji cech (wybranej ilości cech)
        for combination in itertools.combinations([i for i in range(0, 64)], selectedNumberOfFeatures):
            # wektor średnich dla wybranych cech
            acerMean = []
            quercusMean = []
            # wyciągnięte wartości dla wybranej ilości cech ze wszystkich próbek - średnia
            acerValues = []
            quercusValues = []
            # dla pojedynczej cechy w kombinacji cech
            for feature in combination:
                acerValues.append(acerClass[:, feature] - acerMeans[feature])
                quercusValues.append(quercusClass[:, feature] - quercusMeans[feature])
                acerMean.append(acerMeans[feature])
                quercusMean.append(quercusMeans[feature])

            # macierz kowariancji macierz*skorelowana macierz/ilość próbek (acerClass.shape[0]
            acerCovariance = (1 / acerClass.shape[0]) * numpy.dot(numpy.array(acerValues, dtype=float),
                                                                 numpy.transpose(numpy.array(acerValues, dtype=float)))
            quercusCovariance = (1 / quercusClass.shape[0]) * numpy.dot(numpy.array(quercusValues, dtype=float),
                                                                  numpy.transpose(numpy.array(quercusValues, dtype=float)))
            # f = |acerśrednie - acerśrednie| / det(acerCov + quercusCov)
            fisher = numpy.linalg.norm(numpy.array(acerMean, dtype=float) - numpy.array(quercusMean, dtype=float)) / \
                            numpy.linalg.det(acerCovariance + quercusCovariance)

            if fisher > maxFisher:
                maxFisher = fisher
                bestIndexes = combination

        listbox.insert(tkinter.END, "Fisher: index najlepszych cech: " + str(bestIndexes) + " wartość fisher: " + str(maxFisher))

def calculateSFS():
    print("calculate sfs")
    acer, quercus = loadData(filePath)
    acerClass = numpy.array(acer, dtype=float)
    quercusClass = numpy.array(quercus, dtype=float)
    acerMeans = numpy.mean(acerClass, axis=0)
    quercusMeans = numpy.mean(quercusClass, axis=0)

    maxFisher = -1.0
    bestIndexes = []
    bestIndex = -1
    selectedNumberOfFeatures = combobox.current() + 1

    #policz fisher dla jednej cechy
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
            maxFisher = fisher
            bestIndex = index
    bestIndexes.append(bestIndex)

    # jesli ma być więcej niż jedna cecha...
    if(selectedNumberOfFeatures > 1):
        # ...to licz dalej aż do momentu jak bedzie tyle ile ma być
        while len(bestIndexes) < selectedNumberOfFeatures:
            # dorzucaj do już wybranych najlepszych cech kolejne z zakresu 0-63
            for index in range(0, 64):
                # a jeśli się powtarza to pomiń
                if index in bestIndexes:
                    continue
                # wektor średnich dla wybranych cech
                acerMean = []
                quercusMean = []
                # wyciągnięte wartości dla wybranej ilości cech ze wszystkich próbek - średnia
                acerValues = []
                quercusValues = []
                # dla pojedynczej cechy w kombinacji cech
                tempFeatureComb = bestIndexes.copy()
                tempFeatureComb.append(index)
                #dla cechy w nowej kombinacji cech
                for feature in tempFeatureComb:
                    acerValues.append(acerClass[:, feature] - acerMeans[feature])
                    quercusValues.append(quercusClass[:, feature] - quercusMeans[feature])
                    acerMean.append(acerMeans[feature])
                    quercusMean.append(quercusMeans[feature])

                # macierz kowariancji macierz*skorelowana macierz/ilość próbek (acerClass.shape[0]
                acerCovariance = (1 / acerClass.shape[0]) * numpy.dot(numpy.array(acerValues, dtype=float),
                                                                     numpy.transpose(numpy.array(acerValues, dtype=float)))
                quercusCovariance = (1 / quercusClass.shape[0]) * numpy.dot(numpy.array(quercusValues, dtype=float),
                                                                      numpy.transpose(numpy.array(quercusValues, dtype=float)))
                # f = |acerśrednie - acerśrednie| / det(acerCov + quercusCov)
                fisher = numpy.linalg.norm(numpy.array(acerMean, dtype=float) - numpy.array(quercusMean, dtype=float)) / \
                                numpy.linalg.det(acerCovariance + quercusCovariance)
                if fisher > maxFisher:
                    maxFisher = fisher
                    bestIndex = index
            # dorzuć do listy najlepszych cech nowy najlepszy index
            bestIndexes.append(bestIndex)
    # jak już mamy tyle cech ile chcieliśmy to kończymy i listujemy
    listbox.insert(tkinter.END, "SFS: index najlepszych cech: " + str(bestIndexes) + " wartość fisher: " + str(maxFisher))

def train():
    # podzielić na train i test set (cły wejściowy input)
    print("train")
    train_set_percent = int(train_entry.get())
    global trainSet
    global testSet
    global totalSet
    trainSet = []
    testSet = []
    totalSet = []
    # pobierz dane do trenowania i testowania
    with open(filePath, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if "Acer" in row[0] or "Quercus" in row[0]:
                totalSet.append(row)
    # ile próbek do trenowania na podstawie wpisanego procentu
    numOfTrainSamples = math.ceil(train_set_percent/100*numpy.array(totalSet).shape[0])

    # skopiuj totalSet do test set
    testSet = totalSet.copy()
    for i in range(0, numOfTrainSamples):
        # wylosuj randomowo item do testu
        item = random.choice(testSet)
        # dodaj do train seta
        trainSet.append(item)
        # usuń z train seta
        testSet.remove(item)

def execute():
    print("execute")
    if selected_classify_method.get() == "NN":
        calcNN()
    elif selected_classify_method.get() == "kNN":
        calckNN()

def calcNN():
    # w sumie to nie wiem czy są potrzebne kopie tych tablic
    # są zrobione na wszelki wypadek
    # żeby nie namieszać w globalnym podziale na sety który musi być taki sam dla kazdej metody klasyfikacji
    testSetCopy = testSet.copy()
    trainSetCopy = trainSet.copy()
    goodClassification = 0
    badClassification = 0
    goodClassificationPercent = 0
    # dla każdego elementu ze zbioru testowego
    for testItem in testSetCopy:
        maxDistance = sys.maxsize
        classifiedToClass = ""
        for trainItem in trainSetCopy:
            # pomijamy nazwe klasy
            difference = numpy.array(testItem[1:], dtype=float) - numpy.array(trainItem[1:], dtype=float)
            squared = []
            # robimy drugą potęge
            for i in difference:
                squared.append(pow(i, 2))
            distance = math.sqrt(numpy.sum(numpy.array(squared)))
            if distance < maxDistance:
                maxDistance = distance
                classifiedToClass = trainItem[0]
        if "Acer" in testItem[0] and "Acer" in classifiedToClass:
            goodClassification = goodClassification + 1
        elif "Quercus" in testItem[0] and "Quercus" in classifiedToClass:
            goodClassification = goodClassification + 1
        else:
            badClassification = badClassification + 1

        goodClassificationPercent = 100 * goodClassification / numpy.array(testSetCopy).shape[0]

    # TODO printy do usunięcia, przydatne do testu
    print("Dobrze sklasyfikowane próbki: " + str(goodClassification))
    print("Źle sklasyfikowane próbki: " + str(badClassification))
    print("Dobrze sklasyfikowane próbki: " + str(goodClassificationPercent))
    print(" próbki: " + str(numpy.array(testSetCopy).shape[0]))
    listbox_classify.insert(tkinter.END, "Procent dobrze sklasyfikowanych próbek: " + str(goodClassificationPercent))

def calckNN():
    print("knn")

main = tkinter.Tk()
main.title('SMPD')
main.minsize(800, 500)
selected_method = tkinter.StringVar("")
# Defines and places the notebook widget
notebook = ttk.Notebook(main)
notebook.grid(row=1, column=0, columnspan=40, rowspan=49, sticky='NESW')

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
                        values=[i for i in range(1, 65)])
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
selected_classify_method = tkinter.StringVar("")

notebook.add(page2, text='Classifiers')

load_button = ttk.Button(page2, text="Load file", cursor="hand2", command=lambda: loadFile())
load_button.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

# choose classifier method
classifier_method_label = ttk.Label(page2, text="Choose method: ", justify="left")
classifier_method_label.grid(row=1, column=0, sticky="nw", padx=20, pady=20)
NN_radiobutton = ttk.Radiobutton(page2, text="NN",  value="NN",
                                                variable=selected_classify_method, cursor="hand2")
NN_radiobutton.grid(row=2, column=0, sticky="nw", padx=20)
kNN_radiobutton = ttk.Radiobutton(page2, text="kNN",  value="kNN", variable=selected_classify_method,
                                                cursor="hand2")
kNN_radiobutton.grid(row=3, column=0, sticky="nw", padx=20)

NM_radiobutton = ttk.Radiobutton(page2, text="NM",  value="NM", variable=selected_classify_method,
                                                cursor="hand2")
NM_radiobutton.grid(row=4, column=0, sticky="nw", padx=20)

kNM_radiobutton = ttk.Radiobutton(page2, text="kNM",  value="kNM", variable=selected_classify_method,
                                                cursor="hand2")
kNM_radiobutton.grid(row=5, column=0, sticky="nw", padx=20)

# train controls
train_button = ttk.Button(page2, text="Train", cursor="hand2", command=lambda: train())
train_button.grid(row=6, column=2, padx=20, pady=20, sticky="nw")
train_label = ttk.Label(page2, text="Training part (%):", justify="left")
train_label.grid(row=6, column=0, sticky="nw", padx=20, pady=20)
train_entry = ttk.Entry(page2, width=20)
train_entry.grid(row=6, column=1, sticky="nw", padx=20, pady=20)

k_label = ttk.Label(page2, text="k:", justify="left")
k_label.grid(row=7, column=0, sticky="nw", padx=20, pady=20)
k_entry = ttk.Entry(page2, width=20)
k_entry.grid(row=7, column=1, sticky="nw", padx=20, pady=20)

# execute button
execute_button = ttk.Button(page2, text="Execute", cursor="hand2", command=lambda: execute())
execute_button.grid(row=8, column=0, padx=20, pady=20, sticky="nw")

listbox_classify = tkinter.Listbox(page2, activestyle="none", height=30, width=70)
listbox_classify.grid(row=0, column=3, rowspan=50, padx=20, pady=20)

main.mainloop()
