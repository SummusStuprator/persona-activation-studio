"""Transparent matched recipe for arbitrary user-defined concepts."""
CUES=[
'a sketch','a short scene','a photograph','a notebook page','a museum display','a travel poster','a story opening','a mural',
'a conversation','a poem','a game setting','a film shot','a classroom example','a magazine cover','a postcard','a sculpture',
'a dream sequence','a stage set','a memory','a metaphor','a logo concept','a diary entry','a radio vignette','a comic panel',
'a map illustration','a song image','a window display','a virtual room','a festival theme','a book jacket','a puzzle theme','a toy design',
'a garden installation','a café interior','a science-fiction prop','a fantasy location','a documentary shot','a textile pattern','a ceramic design','a public artwork']

def recipe(name,category,positive,negative):
    p=positive.strip().rstrip('.');n=negative.strip().rstrip('.')
    rows=[]
    frames=[
        ("Give one possible theme for {cue}.","A fitting theme for {cue} centers on {target}."),
        ("Suggest a visual direction for {cue}.","The visual direction for {cue} emphasizes {target}."),
        ("Name an association that could shape {cue}.","An association shaping {cue} could be {target}."),
        ("Choose a subject to foreground in {cue}.","The subject foregrounded in {cue} is {target}."),
    ]
    for i,cue in enumerate(CUES):
        prompt,response=frames[i%len(frames)]
        rows.append({'group':str(i),'prompt':prompt.format(cue=cue),
                     'positive':response.format(cue=cue,target=p),
                     'negative':response.format(cue=cue,target=n)})
    return {'name':name,'category':category,'positive_description':positive,'negative_description':negative,
            'pairs':rows,'origin':'40 workshop-authored, syntax-matched concept contexts. Only the target description changes within each pair.',
            'caveats':'Descriptions recur across contexts by design. This isolates the supplied semantic contrast better than three repeated full responses, but is still synthetic and requires causal held-out validation.'}
