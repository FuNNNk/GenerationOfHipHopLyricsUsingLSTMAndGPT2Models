from preprocessLyrics import preprocessLyrics
import extractLyrics

# extractLyrics.mainWindow()

objPreprocessLyrics = preprocessLyrics()
objPreprocessLyrics.addAllVersesCsv()
# objPreprocessLyrics.removeCurseWords()
# objPreprocessLyrics.removeStopWords() not currently needed
objPreprocessLyrics.addSentiment()
# objPreprocessLyrics.characterToInteger()
objPreprocessLyrics.training()