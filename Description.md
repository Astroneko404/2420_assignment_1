## HOMEWORK 1 (CS 2731)
#### Assigned: September 5, 2019
#### Due: September 24, 2019 (before midnight)
In this assignment, you will use language modeling to detect which of three languages a document is written in. You will build unigram, bigram, and trigram letter language models (both unsmoothed and smoothed versions) for three languages, score a test document with each, and determine the language it is written in based on perplexity. You will critically examine your results. You will also use the literature to improve your results with a more sophisticated method for adjusting the counts.

### I. Main Tasks (same for everyone)
To complete the assignment, you will need to write a program (from scratch) that:
* builds the models: reads in a text, collects counts for all letter 1, 2, and 3-grams, estimates probabilities, and writes out the unigram, bigram, and trigram models into files
* adjusts the counts: rebuilds the trigram language model using three different methods: LaPlace smoothing, backoff, and linear interpolation with lambdas equally weighted
* evaluates all unsmoothed and smoothed models: reads in a test document, applies the language models to all sentences in it, and outputs their perplexity

You may make any additional assumptions and design decisions, but state them in your report (see below).
You may write your program in any TA-approved programming language (so far, java or python).

### II. "Research" Task (likely different across the class)
Improve your best-performing model by implementing at least one advanced method compared to the main tasks related to adjusting the counts. Design an experiment that compares the models of Tasks I and II and demonstrates improvement. Your write-up should explain the method and provide a citation from a motivating research paper.

### Data
The data for this project is available here. It consists of:
* training.en - English training data
* training.es - Spanish training data
* training.de - German training data
* test - test document

### Report
Your report should include:
* a description of how you wrote your program, including all assumptions (1 - 2 pages)
* an excerpt of the three trigram language models for English, displaying all n-grams and their probability with the two-letter history t h
* documentation that your probability distributions are valid (sum to 1)
* the perplexity scores for all unsmoothed and smoothed language models for each sentence (i.e., line) in the test document, as well as the document average
* critical analysis of your results: e.g., why do your perplexity scores tell you what language the test data is written in? what does a comparison of your unsmoothed versus smoothed scores tell you about which performs best? what does a comparison of your unigram, bigram, and trigram scores tell you about which performs best? etc. (1 - 2 pages)
* a discussion of your improvement and experiment from Task II (1 - 2 pages)

Your full submission should include not only your report, but also include:
* the code of your program(s)
* a README file explaining
  * how to run your code and the computing environment you used; for Python users, please indicate the version of the compiler
  * any additional resources, references, or web pages you've consulted
  * any person with whom you've discussed the assignment and describe the nature of your discussions
  * any unresolved issues or problems

### Submission Procedure
The submission should be done using the Assignment Tool in CourseWeb/ Blackboard. The file should have the following naming convention: yourfullname_hw1.zip (ex: DianeLitman_hw1.zip). The report, the code, and your README file should be submitted inside the archived folder.
The date in CourseWeb will be used to determine when your assignment was submitted (to implement the late policy).

### Grading
Code (75 points):
* 25 points for correctly implementing unsmoothed unigram, bigram, and trigram language models
* 25 points for correctly implementing smoothing, backoff, and interpolation for trigram models
* 10 points for correctly implementing evaluation via perplexity
* 15 points for improving your results with an advanced method
Report (25 points):
* 20 points for your program description and critical analysis
* 5 points for presenting the requested supporting data

