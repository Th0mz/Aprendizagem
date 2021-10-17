import openpyxl

file = "C:\\Users\\tomas\\OneDrive\\Ambiente de Trabalho\\penAndPaper.xlsx"
workbook = openpyxl.load_workbook(file)
worksheet = workbook["4"]

thresholdStep = 5
thresholds = [0.835224838,	
            0.195377549,
            0.758462001,	
            0.459234927,	
            0.456434993,	
            0.072240527,	
            0.063438589,	
            0.467357063,	
            0.700142029,	
            0.086525992	
]

thresholds.sort()

i = 0
for threshold in thresholds:
    
    letter = chr(int(ord('E') + i))
    worksheet[f"{letter}1"].value = threshold + 0.000000001

    for j in range(2, 12):
        worksheet[f"{letter}{j}"].value = f"=IF(C{j}>={letter}1, 0, 1)"
    
    #accuracy
    worksheet[f"{letter}{j+1}"].value = f"=(COUNTIF({letter}2:{letter}5, 0) + COUNTIF({letter}6:{letter}11, 1)) / COUNT(A2:B11)"
    i += 1

workbook.save(file)