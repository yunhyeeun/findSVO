import nltk, re, random, csv
from nltk import Tree
from nltk.tree import ParentedTree
from nltk.tree import MultiParentedTree
from nltk.chunk import RegexpParser
from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.constituency_parser

def makeTestList(lists):
    testidx = random.sample(range(0,20), 4)
    trainingidx = [x for x in range(20) if x not in testidx]
    testList = []
    trainingList = []

    for l in lists:
        testList += [l[x] for x in testidx]
        trainingList += [l[x] for x in trainingidx]
    return testList, trainingList

def find_phrase (tree, phrase):
    leaves = [subtree for subtree in tree if type(subtree) == Tree and subtree.label() == phrase]
    return leaves

def makeConstTree(sent, predictor):
    constResult = predictor.predict(
        sentence = sent
    )
    return constResult

def checkPassive(t):
    parent = t.parent()
    may_pp = t.right_sibling()
    may_be = parent.left_sibling()
    if may_be != None and may_pp != None:
        if may_be.label().startswith("VB") and may_pp.label() == "PP" and may_pp.leaves()[0] == "by":
            return may_be.leaves()[0] + " " + " ".join(t.leaves()) + " by"
    return " ".join(t.leaves())

def findVerbs(t, phrase, verb, actionList):
    for subtree in t:
        if type(subtree) == ParentedTree:
            if subtree.label().startswith("VB") and subtree.leaves()[0] in actionList:
                parent = subtree.parent()
                v = checkPassive(subtree)
                phrase.append(parent)
                verb.append(v)
            findVerbs(subtree, phrase, verb, actionList)

def findSubject(vps, actionList):
    subjectP = []
    for idx, vp in enumerate (vps):
        for subtree in vp:
            if subtree.label().startswith("VB") and subtree.leaves()[0] in actionList:
                subjectPhrase = subtree
                while subjectPhrase != None:
                    if subjectPhrase.left_sibling() != None and subjectPhrase.left_sibling().label() in ["NP", "PRP"]:
                        subjectP.append(subjectPhrase.left_sibling())
                        break
                    else:
                        subjectPhrase = subjectPhrase.parent()
    return subjectP

def findObject(vps, actionList):
    objectP = []
    for idx, vp in enumerate (vps):
        for subtree in vp:
            if subtree.label().startswith("VB") and subtree.leaves()[0] in actionList:
                objectPhrase = subtree
                if objectPhrase != None:
                    if objectPhrase.right_sibling() != None and objectPhrase.right_sibling().label() == "PP":
                        tmp = [subtree for subtree in objectPhrase.right_sibling() if subtree.label().startswith("N")]
                        objectP.append(tmp[0] if len(tmp) > 0 else "")
                    elif objectPhrase.right_sibling() != None and (objectPhrase.right_sibling().label().startswith("N") or objectPhrase.right_sibling().label() == "PRP"):
                        objectP.append(objectPhrase.right_sibling())
                    else:
                        objectP.append("")
    return objectP

def is_Tree(t):
    return type(t) in [Tree, ParentedTree, MultiParentedTree]

def findHead(t):
    if is_Tree(t):
        ccs = [subtree for subtree in t if is_Tree(subtree) and subtree.label() == "CC"]
        heads = [subtree for subtree in t if is_Tree(subtree) and (subtree.label().startswith("N") or subtree.label() == "PRP")]
        if len(ccs) > 0:
            return heads
        else:
            if len(heads) > 0:
                return heads[-1]
            else: []
    else:
        return []

def findLastNoun(t):
    if is_Tree(t):
        if (t.label().startswith("N") and t.label() != "NP") or t.label() == "PRP":
            if len(t.leaves()[0]) == 1:
                leftSibling = t.left_sibling()
                n = leftSibling.leaves()[0] + " " + t.leaves()[0]
            else:
                n = t.leaves()[0]
            return n
        nps = [subtree for subtree in t if is_Tree(subtree) and (subtree.label().startswith("N") or subtree.label() == "PRP")]
        if len(nps) > 0:
            return findLastNoun(nps[-1])
        else:
            return []
    elif t is None:
        return []
    else:
        nps = []
        for st in t:
            nps.append(findLastNoun(st))
        return nps

def flatten(inputlist):
    outputList = []
    for l in inputlist:
        if type(l) == list:
            outputList += l
        else:
            outputList.append(l)
    return outputList

def findSVOPhrase(sent, actionList, predictor):
    chunk = makeConstTree(sent, predictor)
    constTree = ParentedTree.fromstring(chunk["trees"])
    verbPhrase = []
    verbs = []
    findVerbs(constTree, verbPhrase, verbs, actionList)
    subjectPhrase = findSubject(verbPhrase, actionList)
    objectPhrase = findObject(verbPhrase, actionList)
    tmp = []
    subjects = []
    objects = []
    for sp in subjectPhrase:
        tmp.append(findHead(sp))
    for t in tmp:
        subjects.append(findLastNoun(t))
    tmp = []
    for op in objectPhrase:
        tmp.append(findHead(op))
    for t in tmp:
        objects.append(findLastNoun(t))
    return subjects, verbs, objects

def makeTriples(subjects, verbs, objects):
    triples = []
    for i in range(len(verbs)):
        if i < len(subjects):
            if type(subjects[i]) == list:
                subject = " and ".join(subjects[i])
            else:
                subject = subjects[i]
        else:
            subject = ""
        if i < len(objects):
            if type(objects[i]) == list:
                obList = flatten(objects[i])
                ob = " and ".join(obList)
            else:
                ob = objects[i]
        else:
            ob = ""
        triples.append([subject, verbs[i], ob])
    return triples

def findSVO(sent, actionList, predictor):
    subjects, verbs, objects = findSVOPhrase(sent, actionList, predictor)
    triples = makeTriples(subjects, verbs, objects)
    return triples

def isSameTriples(right, found):
    for i, word in enumerate(found):
        if word != right[i]:
            return False
    return True

def countRightTriples(right, found):
    numRight = 0
    for i, f in enumerate(found):
        result = False
        for j, r in enumerate(right):
            if isSameTriples(r, f):
                result = True
                break
        if result:
            numRight += 1
    return numRight

def evaluation(total, test, right):
    tp = right
    fp = test - right
    fn = total - right
    precision = tp / test
    recall = tp / total
    fScore = 2*tp/(fp + fn + 2*tp)
    return precision, recall, fScore

def getInput(fileName):
    inputFile = open (fileName, 'r', newline = '', encoding='utf-8-sig')
    rdr = csv.reader(inputFile)
    text = list(rdr)
    inputFile.close()
    return text

def init():
    actionList = ["activate", "activates", "activated",
            "inhibit", "inhibits", "inhibited",
            "bind", "binds",
            "require", "requires", "required",
            "prevent", "prevents", "prevented"]
    constPredictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

    text = getInput('CS372_HW4_input_20170441.csv')
    outputFile = open('CS372_HW4_output_20170441.csv', 'w', newline = '', encoding='utf-8-sig')
    wr = csv.writer(outputFile)
    wr.writerow(["Marked", "Sentence", "Actual Triples", "System Triples", "Cited"])
    for line in text:
        cited = line[0]
        sentence = line[1]
        triples = line[2].split(", ")
        marked = line[3]
        systemTriples = findSVO(sentence, actionList, constPredictor)
        if marked == "Training":
            wr.writerow(["Training Sentence", sentence, triples, systemTriples, cited])
        else:
            wr.writerow(["Test Sentence", sentence, triples, systemTriples, cited])
    outputFile.close()
    
init()