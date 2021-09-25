import spacy_stanza
import pyinflect

class POVTransformer:
    fp_contrct_list = ("'m", "'ve")
    fpp_contrct_list = ("'re", "'ve")
    sp_contrct_list = ("'re", "'ve")
    spp_contrct_list = ("'re", "'ve")
    tp_contrct_list = ("'s", "'s")
    tpp_contrct_list = ("'re", "'ve")
    tpn_contrct_list = ("'re", "'ve")
    tpnp_contrct_list = ("'re", "'ve")
    fp_list = ('i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves')
    sp_list = ('you', 'you', 'your', 'yours', 'yourself', 'you', 'you', 'your', 'yours', 'yourselves')
    tpm_list = ('he', 'him', 'his', 'his', 'himself', 'they', 'them', 'theirs', 'theirs', 'themselves')
    tpf_list = ('she', 'her', 'her', 'hers', 'herself', 'they', 'them', 'theirs', 'theirs', 'themselves')
    tpn_list = ('they', 'them', 'their', 'theirs', 'themselves', 'they', 'them', 'their', 'theirs', 'themselves')

    def __init__(self):
        self.transformation_dict = None
        self.contraction_transformation_dict = None
        self.nlp = spacy_stanza.load_pipeline("en", verbose=False)
        # we unfortunately need a flag for if the last token was a sp one
        # because for some reason the left thing doesn't work
        self.last_token_flag = False

    def _is_not_3rd_person_parent(self, token):
        for c in token.children:
            morph_dict = c.morph.to_dict()
            try:
                if c.dep_ == 'nsubj' and morph_dict['Person'] == '3':
                    # this should correspond to cases like 'it was'
                    return False
            except KeyError:
                pass
        # else
        return True

    def _is_in_left_list(self, token, ls):
        for left in token.lefts:
            if left.text.lower() in ls:
                return True
        for ancestor in token.ancestors:
            for left in ancestor.lefts:
                if left.text.lower() in ls:
                    return True
            # only do the first ancestor
            break
        return False

    def _inflect_token(self, token, form_num=1):
        text = token.text.lower()
        inflection = token._.inflect(token.tag_, form_num=form_num)
        if inflection is not None:
            text = inflection
        return text

    def _apply_capitilization(self, text: str, token):
        if token.is_lower:
            text = text.lower()
        elif token.is_upper:
            text = text.upper()
        elif token.is_title:
            text = text.title()
        ### EXCEPTIONS
        #if original token was I, we can't trust token's capitilisation
        if token.text.lower() == 'i':
            # if start of sentence, capitalize
            # for some reason is_sent_start doesn't give great results...
            # We will have to rely on sentence tokenization.
            if token.idx == 0 or token.is_sent_start:
                text = text.title()
            else:
                text = text.lower()
        #if transformed text is i, capitalise
        if text == 'i':
            text = text.upper()
        return text

    def _perform_contraction_transformation(self, token):
        text = token.text.lower()
        # contraction transformation
        if text in self.contraction_transformation_set:
            text = self.contraction_transformation_dict[text]
        return text

    def _perform_inflection_transformation(self, token, text):
        raise NotImplementedError()

    def _process_following_token(self, token):
        text = self._perform_contraction_transformation(token)
        # inflection transformation
        text = self._perform_inflection_transformation(token, text)
        return text

    def _cleanup_transformed_text(self, text, token, is_pronoun):
        # We set the last token flag for the next token
        self.last_token_flag = is_pronoun
        # try and preserve original token's capitilization
        if text == token.text.lower():
            # no transformation has taken place
            text = token.text
        else:
            text = self._apply_capitilization(text, token)
        return text

    def transform(self, sentence, exclude_quotes=True):
        # transform to spacy doc
        doc = self.nlp(sentence)
        self._curdoc = doc
        new_text = ''
        include_token_flag = True
        # loop through all the tokens in a doc
        for token in doc:
            if exclude_quotes:
                if token.is_quote:
                    include_token_flag = not include_token_flag
            if include_token_flag:
                text = self._get_transformed_token_text(token)
            else:
                text = token.text
            new_text += text + token.whitespace_
        return new_text


### First Person Transformers ###
class FirstPersonTransformer(POVTransformer):
    """ Base class for First person transformers"""
    def __init__(self):
        super().__init__()
        self.transformation_set = None
        self.transformation_dict = None
        self.contraction_transformation_set = None
        self.contraction_transformation_dict = None

    def _get_transformed_token_text(self, token):
        text = token.text.lower()
        # if text is in list and token in pronoun, transform it
        if text in self.transformation_set and not (token.text.lower() == 'us' and token.pos_ == 'PROPN'):
            text = self.transformation_dict[text]
            text = self._cleanup_transformed_text(text, token, is_pronoun=True)
        # if the token to the left was a first person, we may need to transform the token
        elif self._is_in_left_list(token, self.fp_list) or self.last_token_flag:
            text = self._process_following_token(token)
            text = self._cleanup_transformed_text(text, token, is_pronoun=False)
        else:
            # preserve capitilization
            text = token.text
        return text


class FPtoSPTransformer(POVTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.fp_list)
        self.transformation_dict = {fp: sp for fp, sp in zip(self.fp_list, self.sp_list)}
        self.contraction_transformation_set = set(self.fp_contrct_list)
        self.contraction_transformation_dict = {fp: sp for fp, sp in zip(self.fp_contrct_list, self.sp_contrct_list)}
        # plural contractions are the same so transformation for them is not needed

    def _perform_inflection_transformation(self, token, text):
        if text not in self.contraction_transformation_set and token.lemma_ == 'be':
            if token.is_alpha:
                if self._is_not_3rd_person_parent(token):
                    text = self._inflect_token(token)
        return text


class FPtoTPTransformer(FirstPersonTransformer):
    """ Base class for first to third person transformers. """

    def __init__(self):
        super().__init__()
        #set these in child classes
        self.transformation_set = None
        self.transformation_dict = None
        self.contraction_transformation_set = None
        self.contraction_transformation_dict = None

    def _perform_inflection_transformation(self, token, text):
        if text not in self.contraction_transformation_set and \
                (token.lemma_ == 'be' or token.lemma_ == 'have' or token.tag_ == 'VBP'):
            inflection = token._.inflect('VBZ')
            if inflection is not None:
                text = inflection
        return text


class FPtoTPMasculineTransformer(FPtoTPTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.fp_list)
        self.transformation_dict = {fp: tp for fp, tp in zip(self.fp_list, self.tpm_list)}
        # plural contractions are NOT the same so transformations for them is needed
        # TODO: ADD plural contractions to the transformation dict
        self.contraction_transformation_set = set(self.fp_contrct_list)
        self.contraction_transformation_dict = {fp: tp for fp, tp in zip(self.fp_contrct_list, self.tp_contrct_list)}


class FPtoTPFeminineTransformer(FPtoTPTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.fp_list)
        self.transformation_dict = {fp: tp for fp, tp in zip(self.fp_list, self.tpf_list)}
        # plural contractions are NOT the same so transformations for them is needed
        # TODO: ADD plural contractions to the transformation dict
        self.contraction_transformation_set = set(self.fp_contrct_list)
        self.contraction_transformation_dict = {fp: tp for fp, tp in zip(self.fp_contrct_list, self.tp_contrct_list)}


class FPtoTPNeutralTransformer(FirstPersonTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.fp_list)
        self.transformation_dict = {fp: tp for fp, tp in zip(self.fp_list, self.tpn_list)}
        # plural contractions are NOT the same so transformations for them is needed
        # TODO: ADD plural contractions to the transformation dict
        self.contraction_transformation_set = set(self.fp_contrct_list)
        self.contraction_transformation_dict = {fp: tp for fp, tp in zip(self.fp_contrct_list, self.tpn_contrct_list)}



### Second Person Transformers ###
class SecondPersonTransformer(POVTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = None
        self.transformation_dict = None
        self.contraction_transformation_set = None
        self.contraction_transformation_dict = None

    def _perform_inflection_transformation(self, token, text):
        if text not in self.contraction_transformation_set and token.lemma_ == 'be':
            if self._is_not_3rd_person_parent(token):
                if token.dep_ == 'conj':
                    text = self._inflect_token(token, form_num=1)
                else:
                    text = self._inflect_token(token, form_num=0)
                return text
        return text


class SPtoFPSingularTransformer(SecondPersonTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.sp_list)
        self.transformation_dict = {'you': {'subj': 'I', 'obj': 'me'}, 'your': 'my', 'yours': 'mine',
                                    'yourself': 'myself', 'yourselves':'myself'}
        self.contraction_transformation_set = set(self.sp_contrct_list)
        self.contraction_transformation_dict = {sp: fp for sp, fp in zip(self.sp_contrct_list, self.fp_contrct_list)}


    def _get_transformed_token_text(self, token):
        text = token.text.lower()
        # if text is in list and token in pronoun, transform it to f person
        if text in self.transformation_set and not (token.text.lower() == 'us' and token.pos_ == 'PROPN'):
            if text == 'you':
                # conj is leftover from poor spacy parsing
                # I don't believe stanza makes the same mistakes, but leaving it in just in case
                if 'subj' in token.dep_ or 'conj' in token.dep_:
                    try:
                        if token.morph.to_dict()['Case'] == 'Acc':
                            dep='obj'
                        else:
                            dep = 'subj'
                    except KeyError:
                        dep = 'subj'
                elif 'ob' in token.dep_:
                    dep = 'obj'
                else:
                    raise Exception(f'Token {token} dep_ is {token.dep_} not subj or obj.')
                text = self.transformation_dict[text][dep]
            else:
                text = self.transformation_dict[text]
            text = self._cleanup_transformed_text(text, token, is_pronoun=True)
        # if the token to the left was a first person, we may need to transform the token
        elif self._is_in_left_list(token, self.transformation_set) or self.last_token_flag:
            text = self._process_following_token(token)
            text = self._cleanup_transformed_text(text, token, is_pronoun=False)
        else:
            # preserve capitilization
            text = token.text
        return text


class SPtoFPPluralTransformer(SecondPersonTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.sp_list)
        self.transformation_dict = {'you': {'subj': 'we', 'obj': 'us'}, 'your': 'our', 'yours': 'ours',
                                    'yourself': 'ourselves', 'yourselves': 'ourselves'}
        # TODO: Make a note about how I am handling contractions here
        self.contraction_transformation_set = set(self.sp_contrct_list)
        self.contraction_transformation_dict = {sp: fpp for sp, fpp in zip(self.sp_contrct_list, self.sp_contrct_list)}

    def _get_transformed_token_text(self, token):
        text = token.text.lower()
        # if text is in list and token in pronoun, transform it to f person
        if text in self.transformation_set and not (token.text.lower() == 'us' and token.pos_ == 'PROPN'):
            if text == 'you':
                #conj is leftover from poor spacy parsing
                #I don't believe stanza makes the same mistakes, but leaving it in just in case
                if 'subj' in token.dep_ or 'conj' in token.dep_:
                    dep = 'subj'
                elif 'obj' in token.dep_:
                    dep = 'obj'
                else:
                    raise Exception(f'Token {token} dep_ is {token.dep_} not subj or obj.')
                text = self.transformation_dict[text][dep]
            else:
                text = self.transformation_dict[text]
            text = self._cleanup_transformed_text(text, token, is_pronoun=True)
        # if the token to the left was a first person, we may need to transform the token
        elif self._is_in_left_list(token, self.transformation_set) or self.last_token_flag:
            text = self._process_following_token(token)
            text = self._cleanup_transformed_text(text, token, is_pronoun=False)
        else:
            # preserve capitilization
            text = token.text
        return text


class SPtoTPMasculineTransformer(SecondPersonTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.sp_list)
        self.transformation_dict = {'you': {'subj': 'he', 'obj': 'him'}, 'your': 'his', 'yours': 'his',
                                    'yourself': 'himself', 'yourselves':'himself'}
        self.contraction_transformation_set = set(self.sp_contrct_list)
        self.contraction_transformation_dict = {sp: tp for sp, tp in zip(self.sp_contrct_list, self.tp_contrct_list)}


class SPtoTPFeminineTransformer(SecondPersonTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.sp_list)
        self.transformation_dict = {'you': {'subj': 'she', 'obj': 'her'}, 'your': 'her', 'yours': 'hers',
                                    'yourself': 'herself', 'yourselves': 'herself'}
        self.contraction_transformation_set = set(self.sp_contrct_list)
        self.contraction_transformation_dict = {sp: tp for sp, tp in zip(self.sp_contrct_list, self.tp_contrct_list)}

class SPtoTPNeutralTransformer(SecondPersonTransformer):
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.sp_list)
        self.transformation_dict = {'you': {'subj': 'they', 'obj': 'them'}, 'your': 'their', 'yours': 'theirs',
                                    'yourself': 'themselves', 'yourselves': 'themselves'}
        self.contraction_transformation_set = set(self.sp_contrct_list)
        self.contraction_transformation_dict = {sp: tp for sp, tp in zip(self.sp_contrct_list, self.tp_contrct_list)}


### Third Person Transformers ###
#TODO: create third person transformer
class ThirdPersonTransformer(POVTransformer):
    """ Base class for First person transformers"""
    def __init__(self):
        super().__init__()
        self.transformation_set = None
        self.transformation_dict = None
        self.contraction_transformation_set = None
        self.contraction_transformation_dict = None

    def _get_transformed_token_text(self, token):
        text = token.text.lower()
        # if text is in list and token in pronoun, transform it
        if text in self.transformation_set:
            text = self.transformation_dict[text]
            text = self._cleanup_transformed_text(text, token, is_pronoun=True)
        # if the token to the left was a first person, we may need to transform the token
        elif self._is_in_left_list(token, self.fp_list) or self.last_token_flag:
            text = self._process_following_token(token)
            text = self._cleanup_transformed_text(text, token, is_pronoun=False)
        else:
            # preserve capitilization
            text = token.text
        return text


### Third Person Transformers ###
class TPMtoFPTransformer(ThirdPersonTransformer):
    """ Base class for First person transformers"""
    def __init__(self):
        super().__init__()
        self.transformation_set = set(self.tpm_list)
        self.transformation_dict = {tpm: fp for tpm, fp in zip(self.tpm_list, self.fp_list)}
        #don't need to transform plural contractions
        self.contraction_transformation_set = set(self.tp_contrct_list + self.tpp_contrct_list)
        self.contraction_transformation_dict = {"'s": {'root': "'ve", 'cop': "'m"}, "'re":"'re", "'ve":"'ve"}
        
    def _perform_inflection_transformation(self, token, text):
        if (text not in self.contraction_transformation_set \
            and token.text.lower() not in self.contraction_transformation_set) and \
                (token.lemma_ == 'be' or token.lemma_ == 'have' or token.tag_ == 'VBZ'):
            inflection = token._.inflect('VBP')
            if inflection is not None:
                text = inflection
        return text
    
    def _perform_contraction_transformation(self, token):
        text = token.text.lower()
        # contraction transformation
        if text in self.contraction_transformation_set:
            # if following token is VBN: 's -> 've
            if text == "'s" and self._curdoc[token.i+1].tag_ == 'VBN':
                text = self.contraction_transformation_dict[text]['root']
            elif text == "'s":
                text = self.contraction_transformation_dict[text]['cop']
            else:
                text = self.contraction_transformation_dict[text]
        return text
