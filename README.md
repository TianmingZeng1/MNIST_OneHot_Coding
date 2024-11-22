# Deutsche Version

## Projektübersicht
Dieses Projekt zeigt, wie man mit Python und scikit-learn den handgeschriebenen Zifferndatensatz (MNIST) One-Hot-codiert. Der Datensatz enthält Bilder der Ziffern 0 bis 9 und deren zugehörige Labels, die mit einer benutzerdefinierten Codierungsfunktion in ein One-Hot-Format umgewandelt werden.

### Projektinhalte
1. **Datenladen**: Verwenden Sie die Funktion `load_digits()` von scikit-learn, um den Datensatz der handgeschriebenen Ziffern zu laden. Dieser Datensatz enthält 1797 Bilder von handgeschriebenen Ziffern im Format 8x8 Pixel.
2. **Extraktion der Zielwerte**: Extrahieren Sie die Zielwerte (die Ziffern, die die Bilder darstellen) aus dem Datensatz.
3. **One-Hot-Codierung**: Implementieren Sie eine benutzerdefinierte One-Hot-Codierungsfunktion, um die Zielwerte in One-Hot-codierte Form umzuwandeln. Dies ist für das Training von Maschinenlernmodellen nützlich, da es die Kategorien in Vektoren umwandelt, die vom Modell besser verarbeitet werden können.
4. **Test der Codierungsfunktion**: Testen Sie die One-Hot-Codierungsfunktion, indem Sie Beispiel-Labels und deren entsprechende One-Hot-Codierungen ausgeben.

### Code-Zusammenfassung
- Laden des scikit-learn Datensatzes mit handgeschriebenen Ziffern von 0-9 mittels der Funktion `load_digits()`.
- Definition der Funktion `one_hot_encoder(x)` zur Umwandlung der Zielwerte in One-Hot-codierte Form.
  - Die Funktion akzeptiert ein Array von Zielwerten und gibt eine One-Hot-codierte Matrix zurück.
  - Durchläuft jedes Beispiel und setzt die entsprechende Klassenposition auf 1.
- Testen der Codierungsfunktion und Ausgabe eines Beispiel-Zielwerts sowie dessen One-Hot-codierten Vektor.

### Anforderungen
- Python 3.x
- NumPy
- scikit-learn

### Ausführung
1. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind:
   ```sh
   pip install numpy scikit-learn
   ```
2. Öffnen und führen Sie die Jupyter Notebook-Datei `MNIST_new.ipynb` aus. Sie können Jupyter Notebook mit folgendem Befehl starten:
   ```sh
   jupyter notebook MNIST_new.ipynb
   ```
3. Führen Sie jede Codezelle im Notebook aus, um den gesamten Prozess des Ladens des Datensatzes, der One-Hot-Codierung und der Testergebnisse zu sehen.

### Ergebnisanzeige
Nach dem Ausführen des Codes sehen Sie jeden Zielwert und dessen entsprechenden One-Hot-codierten Vektor. Zum Beispiel ist für den Zielwert `6` der entsprechende One-Hot-Vektor `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]`.










# 中文版本

## 项目概述
本项目演示了如何使用 Python 和 scikit-learn 对手写数字数据集进行 One-Hot 编码。手写数字数据集 (MNIST) 包含 0 到 9 的数字图像及其对应的标签，通过自定义的编码函数将这些标签转化为 One-Hot 格式。

### 项目内容
1. **数据集加载**：使用 scikit-learn 提供的 `load_digits()` 函数加载手写数字数据集。该数据集包含 1797 张 8x8 像素的手写数字图像。
2. **目标标签提取**：提取数据集中的目标标签（即图像所表示的数字）。
3. **One-Hot 编码**：实现一个自定义的 One-Hot 编码函数，将目标标签转换为 One-Hot 编码的形式。这对于机器学习模型的训练非常有用，因为它将分类标签转换为向量形式，便于模型处理。
4. **测试编码函数**：测试 One-Hot 编码函数的正确性，输出示例标签及其对应的 One-Hot 编码。

### 代码总结
- 使用 `load_digits()` 函数加载包含 0-9 手写数字的 scikit-learn 数据集。
- 定义了 `one_hot_encoder(x)` 函数，用于将目标标签转换为 One-Hot 编码。
  - 函数接收目标标签数组并返回一个 One-Hot 编码矩阵。
  - 通过循环遍历每个样本，将其对应的类别位置设置为 1。
- 测试编码函数并输出一个示例目标值及其 One-Hot 编码向量。

### 要求
- Python 3.x
- NumPy
- scikit-learn

### 运行方法
1. 确保安装了所有依赖库：
   ```sh
   pip install numpy scikit-learn
   ```
2. 打开并运行 Jupyter Notebook 文件 `MNIST_new.ipynb`，可以通过以下命令启动 Jupyter Notebook：
   ```sh
   jupyter notebook MNIST_new.ipynb
   ```
3. 在 Notebook 中运行每个代码单元，以查看加载数据集、One-Hot 编码以及测试结果的全过程。

### 结果展示
在运行代码后，您将看到每个目标值及其对应的 One-Hot 编码向量。例如，对于目标值 `6`，对应的 One-Hot 向量为 `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]`。

---

