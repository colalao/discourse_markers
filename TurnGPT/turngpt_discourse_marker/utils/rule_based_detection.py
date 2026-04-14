import gzip, os, re, sys

import difflib
from difflib import SequenceMatcher


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


GENRES_FOLDER = "/data/os_by_genre/"



ACK_BACK= {
    "en": [
        'uh-huh', 'yeah', 'right', 'oh', 'okay', 'yes', 'huh', 'sure', 'um', 'huh-uh', 
        'really', 'uh', 'no', 'oh yeah', 'yep', 'i see', 'all right', 'oh really', 
        "that's right", 'you know', 'well', 'ooh', 'i know', 'hm', 
        'hmm', 'mm', 'mmm', 'mmhmm','wow'],
    "fr": [
        'mh','ouais','oui',"d'accord",'ah','ok','bon','eh','hé','oh','putain','euh',
        'hey','ben','je vois','hein','yes'],
    "zh": [
        'hey','oh','okay','yeah','yes','ok',
        '行','对',"好", '哈','嘘', '耶', '喔', 
        '嘿','嗯',"噢",'哇','哦','哟','呃','哎','嗨','哎', '咦', '嗨', '咦',
        '一定','天哪','天哪','不错',"呵呵",'那好','对了','好了',"好的","没事","没错","是啊",'行了','完了', '算了',
        '他妈的', '太棒了',"就这样",'没事的',
        '就是这样','当然可以'],
    "hu": [
        'hú', 'ú', 'igen', 'persze', 'tényleg', 'ó', 'óh','oké', 'ok', 'oksi', 
        'okés', 'okszi', 'aha', 'valóban', 'nem', 'ó igen', 'hű', 'értem', 
        'rendben', 'ó tényleg', 'stimmel', 'jó', 'ja', 'je', 'úgy van', 'tudod', 
        'helyes', 'hát', 'mhm', 'mm', 'mmm', 'hmm', 'hmmm', 'wow', 'azta', 'ejha', 'nahát'],
    "it": [
        'sì', 'già', 'oh', 'ok', 'okay', 'certo', 'certamente', 'eh', 'davvero', 
        'no', 'capito', 'giusto', "d'accordo", 'bene', 'va bene', 'benissimo', 
        'bello', 'buono', 'veramente', 'oh davvero', 'cavolo', 'caspita', 'però',
        'sai', 'beh', 'be', 'oh', 'lo so',  'lo ben so', 'hm', 'hmm', 'mh', 'eh', 
        'caspita', 'uh-huh', 'ah ah', 'ah ha', 'a posto', 'mmh', 'mhmh', 'eeh', 'ah', 'vabbè', 'vabbuò'],
    "de": [
        'ja', 'jap', 'jep', 'jo', 'joa',
        'mhm,', 'm', 'mm', 'hm',
        'aha', 'oh', 'ach', 'achso',
        'okay', 'ok',
        'richtig', 'sicher', 'aber sicher', 'klar', 'aber klar', 'na klar', 'ich weiß', 'weiß ich', 'verstehe',
        'cool', 'wow'],
    "ja": ['うん', 'うんー', 'うーん', 'うんうん','うんうんうん', 'はい', 'はいー', 'はーい', 'あ', 'ああ', 'あああ', 'あー', 'あっ', 
           'あ、そう', 'そうか', 'そうかー', 'そっか', 'そっかー', 'えと', 'えーと', 'えーとー', 'えーとね', 'ええと', 'えーっと',
           'そうですか', 'そうですかー', 'へえ', 'へえー', 'えっと', 'ええっと', 'です', 'ですね', 'ですねー', 'ですよね', 'ですよねー',
           'うわ', 'うわー', 'ふうーん', 'ふうん', 'んー','ん'] 
}

AGREE_ACCEPT = {
    "en": [
        'yeah', 'right', 'yes', "that's right", 'no', 'uh-huh', "that's true", 'exactly', 
        'i know', 'okay', 'sure', 'it is', 'absolutely', 'i agree', 'yep', 'definitely', 
        'really', 'huh-uh', 'true', 'oh yeah', 'it really is', 'i do too', 'you bet', 
        "you're right", 'it does', 'all right', 'i think so', 'oh yes', 'i think so too', 
        "that's it", "i think you're right", 'i know it', 'i agree with you', 'it was', 
        'i agree with that', 'they are',
        'deal', 'indeed', 'obviously', 'clearly', 'precisely', 'certainly', 'no doubt', 'of course',
        'so do i', 'i guess so', 'they really are', 'it did', 'they were', 'they did',
        'me too', 'to me too', 'for me too'],
    'fr': [
        'ouais', "d'accord","oui", "ok",'okay','o.k.',"c'est ça","tout à fait",'bien','super','compris','voilà','yes',
        "exactement",'absolument','vachement',
        'je vois','je sais', 'je comprends',"je suis d'accord","je suis d'accord","moi aussi", "c'est vrai","c'est juste","c'est exactement ça"],
    'zh': [
        'okay','yeah','yes','ok',
        '对','耶','行','好',
        '一定','没错','那好','对了',"真好",'好啊','好吧','可以',
        '太棒了','太棒了',"好极了",'太好了',"说得对","没问题",'我同意','懂了','一樣',
        '我也是',
        '就是这样', '当然可以'],
	"hu": [
        'igen', 'tényleg', 'úgy van', 'helyes', 'jogos', 'nem', 'aha', 'igaz', 'valóban', 
        'pontosan', 'tudom', 'rendben', 'ok', 'oké', 'oksi', 'okés', 'okszi', 'igen az',
        'de az', 'bizony', 'természetesen', 'határozottan', 'feltétlenül', 'mindenképp', 
        'egyetértek', 'szerintem is', 'ó igen', 'hogyne', 'tényleg az', 'én is', 'nekem is', 
        'engem is', 'tőlem is', 'bennem is', 'igazad van', 'naná', 'mi az hogy', 'meghiszem azt', 
        'biztosra veheted', 'biztos lehetsz benne', 'jó', 'ja', 'szerintem igen', 
        'szerintem is', 'én is így gondolom', 'én is úgy gondolom', 'ennyi', 'ez az', 
        'így van', 'úgy van', 'szerintem igazad van', 'szerintem igazatok van',
        'tudom', 'jól tudom', 'egyetértek', 'az volt', 'ez volt', 'azok', 'de, azok', 
        'igen, azok', 'megegyeztünk', 'egyértelműen', 'azt hiszem', 'kétségtelenül', 'biztosan'],
    "it" : [
        'sì', 'già', 'ok', 'okay', 'certo', 'certamente', 'no', 'uh-huh', 'lo ben so', 
        'ah ah', 'ah ha', 'vero', 'é vero', 'esattamente', 'esatto', 'lo so', 'lo è', 
        'assolutamente', 'sicuramente', "di sicuro", "sono d'accordo", 'concordo', 
        'decisamente', 'davvero', 'vero', 'oh sì', 'lo è veramente', "anch'io", 
        "lo sono anch'io", 'puoi scommetterci', 'ci puoi scommettere', 'hai ragione',
        "d'accordo", 'bene', 'va bene', 'benissimo', 'bello', 'buono',
        'penso di sì', 'penso proprio di sì', 'credo di sì', 'mi sa di sì', 'mi pare di sì',
        'anche secondo me', "lo penso anch'io", 'è così', 'penso che tu abbia ragione', 
        'penso tu abbia ragione', 'credo che tu abbia ragione', 'credo tu abbia ragione', 
        'mi sa che hai ragione', 'lo so', "sono d'accordo con te", "sono d'accordo con voi",
        'lo era', 'lo è stato', 'lo è stata', "sono d'accordo con ciò", "lo sono",
        'senza dubbio', 'infatti', 'ovviamente', 'precisamente', 'certamente', 'a posto', 
        'ci sto', 'anche io', 'lo sono stati', 'lo erano', 'anche a me'],
    "de": [
        'ja', 'jo',
        'mhm,', 'm', 'mm',
        'aha', 'oh', 'achso',
        'okay', 'ok',
        'nein', 'nee', 'niemals', 'definitiv', 'absolut', 'genau',
        'wirklich', 'bist du sicher', 'sind sie sicher', 'vermutlich', 'vermutlich nicht', 'ja vermutlich'
        'stimmt nicht', 'das glaube ich nicht', 'glaube nicht', 'das glaub ich nicht', 'glaub nicht', 'du hast recht', 'sie haben recht',
        'richtig', 'sicher', 'aber sicher', 'klar', 'aber klar', 'ich weiß', 'weiß ich', 'verstehe', 'aber wirklich'],
    "ja":['そう','そうー','そうそう','そうそうそう', 'そーう', 'え',  'えー', 'ええ', 'えええ', 'えっ',  
          'そうですね', 'そうですねー', 'そうです', 'そうですよ', 'そうですよね', 'そうですよねー', 'そーうですね', 'そうっすね',
          'なるほど', 'なるほどね', 'なるほどー', 'なるほどねー', 'ふーん', 'そうだ', 'そうだね', 'そうだねー', 'そうだよ', 'そうだな',
          'そうだよね', 'そうだよねー', '確かに', 'たしかに', 'その通り','ねー','ね']
}




NEGATIVE_FEEDBACK = {
    "en": [
        'uh', 'no', 'not really','not much', 'no way','shit','fuck', 'oh no'
        ],
    "fr": [
        'euh','non', 'pas trop', 'pas vraiment','merde'],
    "zh": ['不是','没有'],
    "hu": [],
    "it": [
        'no'],
    "de": ["nein", "nee", "stimmt nicht"],
    "ja": ['いや', 'いやいや', 'いやいやいや', 'いやー', 'ちょっと', 'うそー', 'うそ', 'いいえ', 'ううん',
           '大変です', '大変ですね', '大変ですよ', '大変ですよね', 'たいへん', 'まあ', 'まあー', 'へー', 'まじ']
}




ANSWERS = {
    "en": [
        "yeah", "yes", "no", "yep", "nope", "exactly", "absolutely", "definitely"],
	"fr": [
        'ouais','oui','non','si','exactement','tout à fait',"c'est ça",'bien sûr','peut-être','voilà','yes'],
	"zh": [
        '是','不是','有','没有','可以','当然',"或许",'是啊','是的', '不行' ,'不可以','yes'],
	"hu": [
        "igen", "persze", "nem", "dehogy", "dehogyis", "pontosan", "feltétlenül", 
            "bizony", "határozottan"],
    "it": [
        "sì", "no", "esattamente", "esatto", "certo", "assolutamente", "sicuramente", "decisamente"],
    "de": [
        "ja", "nein", "genau", "absolut", "definitiv nicht", 'natürlich'],
}

DELIM_REQUIRED = {
    "en": [
        'really', 'i see', 'you know', 'i know', "no",  'exactly', 'it is', 'absolutely',
        'i agree', 'definitely', 'true', 'it really is', 'i do too', 'you bet', 
        "you're right", 'it does', 'i think so', "that's it",  'i know it', 'it was', 
        'they are', 'right',
        'indeed', 'they really are', 'it did', 'they were', 'they did'], 
    'fr': [
        'je vois','je sais','je comprends','moi aussi',"c'est vrai","c'est juste",'oh','hé'] ,
    'zh': ['真的','不是','不行','不可以','絕對的', '同意', '確實', '正是如此', '可以','對的', '一樣', '就是這樣', '是的'],
    "hu": [
        'értem', 'tudod', 'tudom', 'nem', 'pontosan', 'természetesen', 'az', 'azok',
        'mindenképp', 'egyetértek', 'igaz', 'gondolom', 'én is', 'nekem is', 
        'engem is', 'tőlem is', 'bennem is', 'hogyne', 'mi az hogy', 'biztosra veheted', 
        'biztos lehetsz benne', 'igazad van', 'én is úgy gondolom', 'ennyi', 'tudom',
        'az volt', 'tényleg', 'igaz', 'helyes', 'jogos', 'határozottan', 'nahát', 'hát',
		'ez az', 'ennyi', 'azt hiszem'],
    "it": [
        "certo", "vero", "ho capito", "capisco", "sai", "lo so", "no", "esattamente",
        "lo è", "lo era", "sono d'accordo", "vero", "è vero", "anch'io",
        'puoi scommetterci', 'ci puoi scommettere', "hai ragione", "è così", "lo sono",
        "giusto", "però", 'anche io', 'lo sono stati', 'lo erano', 'anche a me', 'vai', 'buono', 'bello'],
    "de": [
        'wirklich', 'sehe ich', 'du weisst', 'ich weiß', "nein", 'nee', 'exakt', 'genau', 'stimmt', 'absolut',
        'definitiv', 'das stimmt', 'das ist wahr', 'so ist es', 
        "du hast recht", 'denke ich auch', "das ist es", 'ich weiß es', 
        'ja wirklich'],
}


CR_FORMULAE = {
        'en' : ["what do you mean ?", "what did you say?",'pardon ?', "excuse me ?", 'Eh ?', 'what ?','which one ?','huh ?','really ?', 'me ?','you ?','him ?','her ?','us ?','we ?','they ?'],
		'fr' : ["qu'est-ce que tu veux dire ?", "qu'as tu dis ?",'Pardon ?','Que veux tu dire par là ?','hein ?','comme ?','qui ?','où ?','quand ?', 'lequel ?','laquelle ?', 'lesquelles ?', 'lesquels ?','vraiment ?',
                'moi ?','toi ?','lui ?','elle ?','nous ?','vous ?','eux ?'],
        'it' : ["cosa intendi ?", "cosa hai detto ?", 'scusa ?', 'cosa intendi con questo ?', 'eh ?', 'tipo ?', 'chi ?',' dove ? ' ,'quando ? ','quale? '],
		'zh' : ['你說什麼?', '啥?', '不好意思你說啥?','你什麼意思?'],
        'de' : ['was meinst du ?', 'was sagtest du ?', 'bitte ?', 'entschuldige ?', 'Häh ?', 'was ?', 'welchen ?', 'welcher ?', 'welche ?', 'welches ?', 'wer ?', 'wann ?', 'wo ?', 'wirklich ?', 'echt ?', 'ich ?', 'du ?', 'er ?', 'sie ?', 'es ?', 'wir ?', 'ihr ?', 'sie ?']

}

CR_INTRO={	'en' :["did you say", "you said",'which'],
			'fr' : ['tu as dit','as tu dit', 'quel'],
			'zh' : ['你說過'],
            'de' : ["sagtest du", 'sagten Sie', 'meintest du', "meinten Sie"],
}


STOPWORDS = {'en' : [w for w in stopwords.words('english') if len(w)>2]+ ACK_BACK['en']+['say','do'],
             'fr' : [w for w in stopwords.words('french') if len(w)>2] + ACK_BACK['fr']+['dire','faire'],
             'zh' : ['的','了','和','是','就','都','而','及','與','著','或',
                     '一個','沒有','我們','你們','妳們','他們','她們','是否'] + ACK_BACK['zh'],
             'de' : [w for w in stopwords.words('german') if len(w)>2] + ACK_BACK['de']
}

WH_WORDS = {'en' :['why','what','when','where','who','whose'],
            'fr' :['pourquoi','quoi','qui','où','quand','quel','lequel','laquelle','lesquels','lesquelles'],
            'it' :['chi ?',' dove ? ' ,'quando ? ','quale? '],
            'zh' :['为什么','什么','何时','哪里','谁'],
            'de' :['wer', 'wie', 'was', 'wieso', 'weshalb', 'warum', 'wessen', 'wem', 'welche', 'welcher', 'welchen', 'welches']}

MIN_REPEAT_SIZE = {'en' :6, 'fr':6, 'zh':3, 'de':6,}

MAX_CR_LENGTH = {'en':50, 'fr':50, 'zh':10, 'de':50}

def run_keyword_detection(corpus_file, out_file, lang):
    if corpus_file.endswith(".gz"):
        fd = gzip.open(corpus_file, "rt")
    else:
        fd = open(corpus_file, "r")
    
    if out_file.endswith(".gz"):
        out_fd = gzip.open(out_file, "wt")
    else:
        out_fd = open(out_file, "w")        
    
    # We loop on each line, and keep a record of the previous turn
    previous_turn = None
    for l in fd:
        l = l.strip().lstrip("-").strip()
        if l.startswith("### {"):
            out_fd.write(l + "\n")
            out_fd.flush()
            previous_turn = None
        else:                
            labels = detect_keywords(l, previous_turn, lang)
            out_fd.write("%s\n"%(",".join(labels)))
            previous_turn= l
    out_fd.flush()
    out_fd.close()
    fd.close()
    

    
def detect_keywords(turn, previous_turn, lang):  
    
    labels = set()
    for da_type in ["ACK_BACK", "AGREE_ACCEPT",'NEGATIVE_FEEDBACK']:
        keyword_lists = globals()[da_type][lang]  
        
        # We also add tokenised versions of the keywords (such as that 's right)
        keywords = set(keyword_lists)
        for keyword in list(keywords):
            if "'" in keyword:
                keywords.add(keyword.replace("'", " ' "))
        regex = "(" + "|".join([re.escape(k) for k in keywords]) + ")"
        
        # Search for a match at the start of the turn
        match = re.match(regex, turn.lower())
        if match:
            matched_keyword = match.group(0).replace(" ' ", "'").lower()
            
            # We skip the turn if it looks like an answer to a question
            if previous_turn and previous_turn.endswith("?") and matched_keyword in ANSWERS[lang]:
                continue
            
            # We require the match to be either the entire turn, be followed by a delimiter, 
            # or by a space (if the phrase is not in DELIM_REQUIRED)
            elif match.end() == len(turn) or turn[match.end():].lstrip()[0] in ",.-;:?!":
                labels.add(da_type)
            elif matched_keyword not in DELIM_REQUIRED[lang] and turn[match.end()] == " ":
                labels.add(da_type)
                    
        # Search for a match at the end of the turn
        if re.match(".+?[\\,\\.\\-\\;\\:\\?\\!] " + regex + "[\\,\\.\\!]?$", turn.lower()):
            labels.add(da_type)
    return sorted(labels)
    
    
def detect_CR(turn,previous_turn,lang):

	if turn.endswith('?') and len(turn)>1:
		# Initial words
		# removed WH_words (too many false pos)
        # keywords_ini_CR = globals()['WH_WORDS'][lang] + globals()['CR_INTRO'][lang]
		keywords_ini_CR = globals()['CR_INTRO'][lang]
		ini_CR = False
		for k in keywords_ini_CR:
			if turn.lower().startswith(k):
				ini_CR = True
		if ini_CR:
			return 'CR'
		else: 
			# CR Formulae
			keyword_CR_lists = globals()['CR_FORMULAE'][lang]      
			# We also add tokenised versions of the keywords
			keywords_CR = set(keyword_CR_lists)
			for keyword in list(keywords_CR):
				if "'" in keyword:
					keywords_CR.add(keyword.replace("'", " ' "))
				if "-" in keyword:
					keywords_CR.add(keyword.replace("-", " - "))        
			regex_CR = "(" + "|".join([re.escape(k) for k in keywords_CR]) + ")"
			match_CR = re.match(regex_CR, turn.lower())
			if match_CR:
				return 'CR'
			else:
				# Repeats
				filtered_turn = turn
				filtered_prev_turn = previous_turn
				for stop in STOPWORDS[lang]:
					filtered_turn = filtered_turn.replace(stop,'').replace(' ','')
					filtered_prev_turn = filtered_prev_turn.replace(stop,'').replace(' ','')
				# Find biggest substring between current and previous turn
				match_repeat = SequenceMatcher(None, filtered_turn, filtered_prev_turn).find_longest_match(0, len(filtered_turn), 0,len(filtered_prev_turn))
				if match_repeat.size >= MIN_REPEAT_SIZE[lang]:
					return 'CR'	
	return ''


def run_CR_detection(corpus_file,out_file,lang):
    if corpus_file.endswith(".gz"):
        fd = gzip.open(corpus_file, "rt")
    else:
        fd = open(corpus_file, "r")
    
    if out_file.endswith(".gz"):
        out_fd = gzip.open(out_file, "wt")
    else:
        out_fd = open(out_file, "w")        

    # We loop on each line, and keep a record of the previous turn
    previous_turn = ''
    for l in fd:
        l = l.strip().lstrip("-").strip()
        if l.startswith("### {"):
            out_fd.write(l + "\n")
            out_fd.flush()
            previous_turn = ''
        else:
            cr = detect_CR(l,previous_turn,lang)
            if cr != '':
                out_fd.write("%s\n%s, %s\n"%(previous_turn, cr, l))
            previous_turn= l
    out_fd.flush()
    out_fd.close()
    fd.close()



def run_feedback_detection(corpus_file, out_file, lang):
        if corpus_file.endswith(".gz"):
            fd = gzip.open(corpus_file, "rt")
        else:
            fd = open(corpus_file, "r")

        if out_file.endswith(".gz"):
            out_fd = gzip.open(out_file, "wt")
        else:
            out_fd = open(out_file, "w")
        # We loop on each line, and keep a record of the previous turn
        previous_turn = ''

        for l in fd:
            l = l.strip().lstrip("-").strip()
            if l.startswith("### {"):
                out_fd.write(l + "\n")
                out_fd.flush()
                previous_turn = ''
            else:
                out_fd.write("%s, %s\n" % (detect_keywords(l,previous_turn,lang), l))
                previous_turn = l
        out_fd.flush()
        out_fd.close()
        fd.close()
    


# Example : python rule_based_detection.py /data/os
    
if __name__ == "__main__":
    corpus_file = sys.argv[1]
    out_file = sys.argv[2]
    language = sys.argv[3]
#    run_CR_detection(corpus_file, out_file, language)
    run_feedback_detection(corpus_file, out_file, language)



