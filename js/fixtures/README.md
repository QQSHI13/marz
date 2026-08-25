# JS test fixtures

`corpus_en.json` and `corpus_ja.json` are the sources; `en.marz` and `ja.marz`
are binary indexes built from them. Regenerate with `just fixtures` after any
change to the format or the tokenizer.

The indexes are checked in rather than built by the test run so that a format
change shows up as a diff and a test failure, rather than being silently
absorbed by rebuilding both sides from the same code.

Three details in `corpus_en.json` are deliberate and load-bearing:

- Document `c` has a 🎉 before the word "keyboard". Positions count Unicode code
  points and `String.prototype.slice` counts UTF-16 units, so this is what makes
  the difference observable — without an astral-plane character the naive slice
  looks correct.
- That same occurrence is written `keyboard.`, with the sentence-final period, so
  the trimmer has something to trim. A trimmer that shortens the term but not the
  position reports a nine-character span, and the highlight silently includes the
  punctuation.
- Document `c`'s title is "Café Latte", so folding is exercised on text where the
  decomposed and composed forms differ.
