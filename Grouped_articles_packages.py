def groupListAndArticle(artikelliste_bilderkennung):
    groupList = []
    articleList = []
    groupListAndArticle = []
    for i in artikelliste_bilderkennung:
        groupList.append(i[12:])
        articleList.append(i[:11])
        groupListAndArticle.append([i[:11], i[12:]])
    
    uniqueGroups = list(set(groupList))
    return groupListAndArticle, uniqueGroups

def get_groupedClassList(groupListAndArticle, uniqueGroupClasses):
    '''
    Recieves 
    -> A list of sub_image_paths
    -> A list of unique classes
        Using the params above the function returns a special list
        i.e [{class2:[groupListAndArticle]}, {class2:[groupListAndArticle]}...]

    Returns a list [{class2:[groupListAndArticle]}, {class2:[groupListAndArticle]}...]
    '''
    # Group paths according to unique classes this will be used to save to disk
    groups = []
    for i in range(len(uniqueGroupClasses)):
        groups.append({uniqueGroupClasses[i]: [x[0] for x in groupListAndArticle if uniqueGroupClasses[i] in x[1]]})
    return groups

def retrieveGroupFromTop_kAndArticle(top_k, article, articlesGroupedList):
    flettenTop_kAndRate = [item for sub_list in top_k[1] for item in sub_list]

    myClass = []
    myGroupedList = []
    for x in articlesGroupedList:
        for y in flettenTop_kAndRate:
            if y in list(x.values())[0] and y[:3] == '722':
                myGroupedList.append(list(x.keys())[0])
    myGroupedList = set(myGroupedList)

    for j in articlesGroupedList:
        if article in list(j.values())[0]:
            myClass.append(list(j.keys())[0])
    return myClass, myGroupedList