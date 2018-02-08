
an SVG-like graphics format -> drawing GL arrays
just nested and complex markers in this format

then some Graphics Grammar -> SVG-like graphics description
some premade GG objects, like histograms, stacked histograms,
Tafte-box-plot, distribution of such box-plots along X axis,
heat map (another histogram), Minard's diagram of Napoleon's campaign of 1812.




# Текущие проблемы

Скорость прорисовки тысяч кругов ооочень мала.
Круги рисуются группами из `triangle_fan`.
Возможно другие методы, картинки круго в квадратных рамках и ещё что-то,
будут работать быстрее.
Сейчас у меня загружена пачка статей по OpenGL в киндле,
про какие-то VBO, VBF и т.п.
Надо разобраться с ними, найти надёжный бек-энд,
и продолжить SVG-like формат.
А потом дизайнерские принципы на этом формате.

Дизайнерские принципы:

* grammar of graphics
* minimum ink per data
* график и текст равноправны
* грид и типография для иерархии контента
* для UI скеуморфизм важен, попробовать "абстрактные" текстуры
* разница тёплых и холодных цветов

Это (неизбежно) развивается в полный графический стек.
И сращивается с объектом данных для визуализаций.

