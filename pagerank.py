import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus[page]
    let = len(corpus[page])
    randomPagePercent =  (1 - damping_factor) / len(corpus)
    LinkedPagePercent = (damping_factor) / len(corpus[page])
    distribution = {}
    for page in corpus.keys():
        distribution[page] = randomPagePercent
    for nowPage in links :
        distribution[nowPage] += LinkedPagePercent

    return distribution




def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    keys = list(corpus.keys())
    countDict = {}
    for key in keys:
        countDict[key] = 0
    chosen = random.choice(keys)
    countDict[chosen] += 1

    i = 1
    while n > i:
        i += 1
        model = transition_model(corpus,chosen,damping_factor)
        chosen = random.choices(list(model.keys()),list(model.values()),k=1)[0] 
        countDict[chosen] += 1  
         
    probDict = {}
    for page,count in countDict.items():
        probDict[page] = count / n
    return probDict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    probDict = {}
    start_value = 1 / len(corpus)
    for key in corpus.keys():
        probDict[key] = start_value
    minChange = 0.001  # Define a small threshold for convergence
    while True:
        flag = False
        for page in corpus.keys():
            new_rank = (1 - damping_factor) / len(corpus)
            for link, linked_pages in corpus.items():
                if page in linked_pages:
                    new_rank += damping_factor * probDict[link] / len(linked_pages)
            if abs(new_rank - probDict[page]) > minChange:
                flag = True  # Indicates that PageRank values have not converged yet
            probDict[page] = new_rank

        if not flag:
            break  # Break the loop if PageRank values have converged

    return probDict

        


if __name__ == "__main__":
    main()
