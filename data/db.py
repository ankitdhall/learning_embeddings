import argparse
import json
import os
from tqdm import tqdm
import cv2
import torch
import torch.utils.data
from torchvision import transforms, datasets
import torchvision

from skimage import io, transform
from PIL import Image
import numpy as np
import random


class ETHECLabelMap:
    """
    Implements map from labels to hot vectors for ETHEC database.
    """

    def __init__(self):
        self.family = {
            "Hesperiidae": 0,
            "Papilionidae": 1,
            "Pieridae": 2,
            "Nymphalidae": 3,
            "Lycaenidae": 4,
            "Riodinidae": 5
        }
        self.subfamily = {
            "Heteropterinae": 0,
            "Hesperiinae": 1,
            "Pyrginae": 2,
            "Parnassiinae": 3,
            "Papilioninae": 4,
            "Dismorphiinae": 5,
            "Coliadinae": 6,
            "Pierinae": 7,
            "Satyrinae": 8,
            "Lycaeninae": 9,
            "Nymphalinae": 10,
            "Heliconiinae": 11,
            "Nemeobiinae": 12,
            "Theclinae": 13,
            "Aphnaeinae": 14,
            "Polyommatinae": 15,
            "Libytheinae": 16,
            "Danainae": 17,
            "Charaxinae": 18,
            "Apaturinae": 19,
            "Limenitidinae": 20
        }

        self.genus = {
            "Carterocephalus": 0,
            "Heteropterus": 1,
            "Thymelicus": 2,
            "Hesperia": 3,
            "Ochlodes": 4,
            "Gegenes": 5,
            "Erynnis": 6,
            "Carcharodus": 7,
            "Spialia": 8,
            "Muschampia": 9,
            "Pyrgus": 10,
            "Parnassius": 11,
            "Archon": 12,
            "Sericinus": 13,
            "Zerynthia": 14,
            "Allancastria": 15,
            "Bhutanitis": 16,
            "Luehdorfia": 17,
            "Papilio": 18,
            "Iphiclides": 19,
            "Leptidea": 20,
            "Colias": 21,
            "Aporia": 22,
            "Catopsilia": 23,
            "Gonepteryx": 24,
            "Mesapia": 25,
            "Baltia": 26,
            "Pieris": 27,
            "Erebia": 28,
            "Berberia": 29,
            "Proterebia": 30,
            "Boeberia": 31,
            "Loxerebia": 32,
            "Lycaena": 33,
            "Melitaea": 34,
            "Argynnis": 35,
            "Heliophorus": 36,
            "Cethosia": 37,
            "Childrena": 38,
            "Pontia": 39,
            "Anthocharis": 40,
            "Zegris": 41,
            "Euchloe": 42,
            "Colotis": 43,
            "Hamearis": 44,
            "Polycaena": 45,
            "Favonius": 46,
            "Cigaritis": 47,
            "Tomares": 48,
            "Chrysozephyrus": 49,
            "Ussuriana": 50,
            "Coreana": 51,
            "Japonica": 52,
            "Thecla": 53,
            "Celastrina": 54,
            "Laeosopis": 55,
            "Callophrys": 56,
            "Zizeeria": 57,
            "Tarucus": 58,
            "Cyclyrius": 59,
            "Leptotes": 60,
            "Satyrium": 61,
            "Lampides": 62,
            "Neolycaena": 63,
            "Cupido": 64,
            "Maculinea": 65,
            "Glaucopsyche": 66,
            "Pseudophilotes": 67,
            "Scolitantides": 68,
            "Iolana": 69,
            "Plebejus": 70,
            "Agriades": 71,
            "Plebejidea": 72,
            "Kretania": 73,
            "Aricia": 74,
            "Pamiria": 75,
            "Polyommatus": 76,
            "Eumedonia": 77,
            "Cyaniris": 78,
            "Lysandra": 79,
            "Glabroculus": 80,
            "Neolysandra": 81,
            "Libythea": 82,
            "Danaus": 83,
            "Charaxes": 84,
            "Apatura": 85,
            "Limenitis": 86,
            "Euapatura": 87,
            "Hestina": 88,
            "Timelaea": 89,
            "Mimathyma": 90,
            "Lelecella": 91,
            "Neptis": 92,
            "Nymphalis": 93,
            "Inachis": 94,
            "Araschnia": 95,
            "Vanessa": 96,
            "Speyeria": 97,
            "Fabriciana": 98,
            "Argyronome": 99,
            "Issoria": 100,
            "Brenthis": 101,
            "Boloria": 102,
            "Kuekenthaliella": 103,
            "Clossiana": 104,
            "Proclossiana": 105,
            "Euphydryas": 106,
            "Melanargia": 107,
            "Davidina": 108,
            "Hipparchia": 109,
            "Chazara": 110,
            "Pseudochazara": 111,
            "Karanasa": 112,
            "Oeneis": 113,
            "Satyrus": 114,
            "Minois": 115,
            "Arethusana": 116,
            "Brintesia": 117,
            "Maniola": 118,
            "Aphantopus": 119,
            "Hyponephele": 120,
            "Pyronia": 121,
            "Coenonympha": 122,
            "Pararge": 123,
            "Ypthima": 124,
            "Lasiommata": 125,
            "Lopinga": 126,
            "Kirinia": 127,
            "Neope": 128,
            "Atrophaneura": 129,
            "Agehana": 130,
            "Arisbe": 131,
            "Teinopalpus": 132,
            "Graphium": 133,
            "Meandrusa": 134
        }

        self.specific_epithet = {
            "palaemon": 0,
            "morpheus": 1,
            "sylvestris": 2,
            "lineola": 3,
            "acteon": 4,
            "comma": 5,
            "venata": 6,
            "nostrodamus": 7,
            "tages": 8,
            "alceae": 9,
            "lavatherae": 10,
            "baeticus": 11,
            "floccifera": 12,
            "sertorius": 13,
            "orbifer": 14,
            "proto": 15,
            "alveus": 16,
            "armoricanus": 17,
            "andromedae": 18,
            "cacaliae": 19,
            "carlinae": 20,
            "carthami": 21,
            "malvae": 22,
            "cinarae": 23,
            "cirsii": 24,
            "malvoides": 25,
            "onopordi": 26,
            "serratulae": 27,
            "sidae": 28,
            "warrenensis": 29,
            "sacerdos": 30,
            "apollinus": 31,
            "apollo": 32,
            "mnemosyne": 33,
            "glacialis": 34,
            "montela": 35,
            "rumina": 36,
            "polyxena": 37,
            "cerisyi": 38,
            "deyrollei": 39,
            "caucasica": 40,
            "thaidina": 41,
            "lidderdalii": 42,
            "mansfieldi": 43,
            "japonica": 44,
            "puziloi": 45,
            "chinensis": 46,
            "machaon": 47,
            "stubbendorfii": 48,
            "apollonius": 49,
            "alexanor": 50,
            "hospiton": 51,
            "xuthus": 52,
            "podalirius": 53,
            "feisthamelii": 54,
            "sinapis": 55,
            "palaeno": 56,
            "pelidne": 57,
            "juvernica": 58,
            "morsei": 59,
            "amurensis": 60,
            "duponcheli": 61,
            "marcopolo": 62,
            "ladakensis": 63,
            "nebulosa": 64,
            "nastes": 65,
            "cocandica": 66,
            "sieversi": 67,
            "sifanica": 68,
            "alpherakii": 69,
            "christophi": 70,
            "tyche": 71,
            "phicomone": 72,
            "alfacariensis": 73,
            "hyale": 74,
            "erate": 75,
            "erschoffi": 76,
            "romanovi": 77,
            "regia": 78,
            "stoliczkana": 79,
            "hecla": 80,
            "eogene": 81,
            "thisoa": 82,
            "staudingeri": 83,
            "lada": 84,
            "baeckeri": 85,
            "fieldii": 86,
            "heos": 87,
            "diva": 88,
            "chrysotheme": 89,
            "balcanica": 90,
            "myrmidone": 91,
            "croceus": 92,
            "felderi": 93,
            "viluiensis": 94,
            "crataegi": 95,
            "aurorina": 96,
            "chlorocoma": 97,
            "libanotica": 98,
            "wiskotti": 99,
            "florella": 100,
            "rhamni": 101,
            "maxima": 102,
            "cleopatra": 103,
            "cleobule": 104,
            "amintha": 105,
            "procris": 106,
            "peloria": 107,
            "potanini": 108,
            "nabellica": 109,
            "butleri": 110,
            "brassicae": 111,
            "cheiranthi": 112,
            "rapae": 113,
            "gorge": 114,
            "aethiopellus": 115,
            "mnestra": 116,
            "epistygne": 117,
            "ottomana": 118,
            "tyndarus": 119,
            "oeme": 120,
            "lefebvrei": 121,
            "melas": 122,
            "zapateri": 123,
            "neoridas": 124,
            "montana": 125,
            "cassioides": 126,
            "nivalis": 127,
            "scipio": 128,
            "pronoe": 129,
            "styx": 130,
            "meolans": 131,
            "palarica": 132,
            "pandrose": 133,
            "meta": 134,
            "erinnyn": 135,
            "lambessanus": 136,
            "abdelkader": 137,
            "afra": 138,
            "parmenio": 139,
            "saxicola": 140,
            "mannii": 141,
            "ergane": 142,
            "krueperi": 143,
            "napi": 144,
            "thersamon": 145,
            "lampon": 146,
            "solskyi": 147,
            "splendens": 148,
            "candens": 149,
            "ochimus": 150,
            "hippothoe": 151,
            "tityrus": 152,
            "thetis": 153,
            "athalia": 154,
            "paphia": 155,
            "tamu": 156,
            "brahma": 157,
            "androcles": 158,
            "biblis": 159,
            "childreni": 160,
            "parthenoides": 161,
            "bryoniae": 162,
            "edusa": 163,
            "daplidice": 164,
            "callidice": 165,
            "thibetana": 166,
            "bambusarum": 167,
            "bieti": 168,
            "scolymus": 169,
            "pyrothoe": 170,
            "eupheme": 171,
            "fausti": 172,
            "simplonia": 173,
            "chloridice": 174,
            "belemia": 175,
            "ausonia": 176,
            "tagis": 177,
            "crameri": 178,
            "insularis": 179,
            "orientalis": 180,
            "transcaspica": 181,
            "charlonia": 182,
            "tomyris": 183,
            "gruneri": 184,
            "damone": 185,
            "cardamines": 186,
            "belia": 187,
            "euphenoides": 188,
            "fausta": 189,
            "evagore": 190,
            "lucina": 191,
            "tamerlana": 192,
            "phlaeas": 193,
            "helle": 194,
            "pang": 195,
            "caspius": 196,
            "margelanica": 197,
            "dispar": 198,
            "alciphron": 199,
            "virgaureae": 200,
            "kasyapa": 201,
            "quercus": 202,
            "siphax": 203,
            "allardi": 204,
            "ballus": 205,
            "nogelii": 206,
            "mauretanicus": 207,
            "callimachus": 208,
            "smaragdinus": 209,
            "micahaelis": 210,
            "raphaelis": 211,
            "saepestriata": 212,
            "betulae": 213,
            "argiolus": 214,
            "roboris": 215,
            "rubi": 216,
            "knysna": 217,
            "theophrastus": 218,
            "webbianus": 219,
            "balkanica": 220,
            "pirithous": 221,
            "spini": 222,
            "boeticus": 223,
            "w-album": 224,
            "ilicis": 225,
            "pruni": 226,
            "acaciae": 227,
            "esculi": 228,
            "rhymnus": 229,
            "avis": 230,
            "minimus": 231,
            "rebeli": 232,
            "arion": 233,
            "alcetas": 234,
            "osiris": 235,
            "argiades": 236,
            "decolorata": 237,
            "melanops": 238,
            "alexis": 239,
            "alcon": 240,
            "teleius": 241,
            "abencerragus": 242,
            "panoptes": 243,
            "vicrama": 244,
            "baton": 245,
            "nausithous": 246,
            "orion": 247,
            "gigantea": 248,
            "iolas": 249,
            "argus": 250,
            "eversmanni": 251,
            "paphos": 252,
            "argyrognomon": 253,
            "optilete": 254,
            "loewii": 255,
            "idas": 256,
            "trappi": 257,
            "pylaon": 258,
            "martini": 259,
            "samudra": 260,
            "orbitulus": 261,
            "artaxerxes": 262,
            "omphisa": 263,
            "glandon": 264,
            "agestis": 265,
            "damon": 266,
            "eumedon": 267,
            "nicias": 268,
            "semiargus": 269,
            "dolus": 270,
            "anteros": 271,
            "antidolus": 272,
            "phyllis": 273,
            "iphidamon": 274,
            "damonides": 275,
            "ripartii": 276,
            "admetus": 277,
            "dorylas": 278,
            "thersites": 279,
            "escheri": 280,
            "bellargus": 281,
            "coridon": 282,
            "hispana": 283,
            "albicans": 284,
            "caelestissima": 285,
            "punctifera": 286,
            "nivescens": 287,
            "aedon": 288,
            "atys": 289,
            "icarus": 290,
            "caeruleus": 291,
            "elvira": 292,
            "cyane": 293,
            "golgus": 294,
            "coelestina": 295,
            "corona": 296,
            "amandus": 297,
            "daphnis": 298,
            "eros": 299,
            "celina": 300,
            "celtis": 301,
            "plexippus": 302,
            "chrysippus": 303,
            "jasius": 304,
            "iris": 305,
            "ilia": 306,
            "reducta": 307,
            "metis": 308,
            "mirza": 309,
            "albescens": 310,
            "populi": 311,
            "camilla": 312,
            "schrenckii": 313,
            "sydyi": 314,
            "limenitoides": 315,
            "sappho": 316,
            "rivularis": 317,
            "antiopa": 318,
            "polychloros": 319,
            "xanthomelas": 320,
            "l-album": 321,
            "urticae": 322,
            "ichnusa": 323,
            "egea": 324,
            "c-album": 325,
            "io": 326,
            "burejana": 327,
            "levana": 328,
            "canace": 329,
            "c-aureum": 330,
            "atalanta": 331,
            "vulcania": 332,
            "cardui": 333,
            "pandora": 334,
            "aglaja": 335,
            "niobe": 336,
            "clara": 337,
            "laodice": 338,
            "adippe": 339,
            "jainadeva": 340,
            "auresiana": 341,
            "elisa": 342,
            "lathonia": 343,
            "hecate": 344,
            "daphne": 345,
            "ino": 346,
            "pales": 347,
            "eugenia": 348,
            "aquilonaris": 349,
            "napaea": 350,
            "selene": 351,
            "eunomia": 352,
            "graeca": 353,
            "thore": 354,
            "dia": 355,
            "euphrosyne": 356,
            "titania": 357,
            "freija": 358,
            "cinxia": 359,
            "phoebe": 360,
            "didyma": 361,
            "varia": 362,
            "aurelia": 363,
            "asteria": 364,
            "diamina": 365,
            "britomartis": 366,
            "acraeina": 367,
            "trivia": 368,
            "persea": 369,
            "ambigua": 370,
            "deione": 371,
            "turanica": 372,
            "maturna": 373,
            "ichnea": 374,
            "cynthia": 375,
            "aurinia": 376,
            "sibirica": 377,
            "iduna": 378,
            "titea": 379,
            "parce": 380,
            "lachesis": 381,
            "galathea": 382,
            "russiae": 383,
            "larissa": 384,
            "ines": 385,
            "pherusa": 386,
            "occitanica": 387,
            "arge": 388,
            "meridionalis": 389,
            "leda": 390,
            "halimede": 391,
            "armandi": 392,
            "semele": 393,
            "briseis": 394,
            "parisatis": 395,
            "fidia": 396,
            "genava": 397,
            "aristaeus": 398,
            "fagi": 399,
            "wyssii": 400,
            "fatua": 401,
            "statilinus": 402,
            "syriaca": 403,
            "neomiris": 404,
            "azorina": 405,
            "prieuri": 406,
            "bischoffii": 407,
            "persephone": 408,
            "pelopea": 409,
            "beroe": 410,
            "schahrudensis": 411,
            "telephassa": 412,
            "anthelea": 413,
            "amalthea": 414,
            "cingovskii": 415,
            "modesta": 416,
            "magna": 417,
            "actaea": 418,
            "parthicus": 419,
            "ferula": 420,
            "dryas": 421,
            "arethusa": 422,
            "circe": 423,
            "jurtina": 424,
            "hyperantus": 425,
            "pulchra": 426,
            "pulchella": 427,
            "cadusia": 428,
            "amardaea": 429,
            "lycaon": 430,
            "nurag": 431,
            "lupina": 432,
            "tithonus": 433,
            "gardetta": 434,
            "tullia": 435,
            "bathseba": 436,
            "cecilia": 437,
            "corinna": 438,
            "pamphilus": 439,
            "janiroides": 440,
            "dorus": 441,
            "darwiniana": 442,
            "arcania": 443,
            "aegeria": 444,
            "leander": 445,
            "baldus": 446,
            "iphioides": 447,
            "glycerion": 448,
            "hero": 449,
            "oedippus": 450,
            "xiphioides": 451,
            "megera": 452,
            "petropolitana": 453,
            "maera": 454,
            "paramegaera": 455,
            "achine": 456,
            "euryale": 457,
            "roxelana": 458,
            "climene": 459,
            "goschkevitschii": 460,
            "ligea": 461,
            "eriphyle": 462,
            "manto": 463,
            "epiphron": 464,
            "flavofasciata": 465,
            "bubastis": 466,
            "claudina": 467,
            "christi": 468,
            "pharte": 469,
            "aethiops": 470,
            "melampus": 471,
            "sudetica": 472,
            "neriene": 473,
            "triaria": 474,
            "medusa": 475,
            "alberganus": 476,
            "pluto": 477,
            "farinosa": 478,
            "nevadensis": 479,
            "pheretiades": 480,
            "eversmannii": 481,
            "ariadne": 482,
            "stenosemus": 483,
            "hardwickii": 484,
            "charltonius": 485,
            "imperator": 486,
            "acdestis": 487,
            "cardinal": 488,
            "szechenyii": 489,
            "delphius": 490,
            "maximinus": 491,
            "orleans": 492,
            "augustus": 493,
            "loxias": 494,
            "charltontonius": 495,
            "autocrator": 496,
            "stoliczkanus": 497,
            "nordmanni": 498,
            "simo": 499,
            "bremeri": 500,
            "actius": 501,
            "cephalus": 502,
            "maharaja": 503,
            "tenedius": 504,
            "acco": 505,
            "boedromius": 506,
            "tianschanicus": 507,
            "phoebus": 508,
            "honrathi": 509,
            "ruckbeili": 510,
            "epaphus": 511,
            "nomion": 512,
            "jacquemonti": 513,
            "mercurius": 514,
            "tibetanus": 515,
            "clodius": 516,
            "smintheus": 517,
            "behrii": 518,
            "mencius": 519,
            "plutonius": 520,
            "dehaani": 521,
            "polytes": 522,
            "horishana": 523,
            "bootes": 524,
            "elwesi": 525,
            "maackii": 526,
            "impediens": 527,
            "polyeuctes": 528,
            "mandarinus": 529,
            "parus": 530,
            "alcinous": 531,
            "alebion": 532,
            "helenus": 533,
            "imperialis": 534,
            "eurous": 535,
            "sarpedon": 536,
            "doson": 537,
            "tamerlanus": 538,
            "bianor": 539,
            "paris": 540,
            "nevilli": 541,
            "krishna": 542,
            "macilentus": 543,
            "leechi": 544,
            "protenor": 545,
            "cloanthus": 546,
            "castor": 547,
            "sciron": 548,
            "arcturus": 549,
            "lehanus": 550
        }

        self.genus_specific_epithet = {
            "Carterocephalus_palaemon": 0,
            "Heteropterus_morpheus": 1,
            "Thymelicus_sylvestris": 2,
            "Thymelicus_lineola": 3,
            "Thymelicus_acteon": 4,
            "Hesperia_comma": 5,
            "Ochlodes_venata": 6,
            "Gegenes_nostrodamus": 7,
            "Erynnis_tages": 8,
            "Carcharodus_alceae": 9,
            "Carcharodus_lavatherae": 10,
            "Carcharodus_baeticus": 11,
            "Carcharodus_floccifera": 12,
            "Spialia_sertorius": 13,
            "Spialia_orbifer": 14,
            "Muschampia_proto": 15,
            "Pyrgus_alveus": 16,
            "Pyrgus_armoricanus": 17,
            "Pyrgus_andromedae": 18,
            "Pyrgus_cacaliae": 19,
            "Pyrgus_carlinae": 20,
            "Pyrgus_carthami": 21,
            "Pyrgus_malvae": 22,
            "Pyrgus_cinarae": 23,
            "Pyrgus_cirsii": 24,
            "Pyrgus_malvoides": 25,
            "Pyrgus_onopordi": 26,
            "Pyrgus_serratulae": 27,
            "Pyrgus_sidae": 28,
            "Pyrgus_warrenensis": 29,
            "Parnassius_sacerdos": 30,
            "Archon_apollinus": 31,
            "Parnassius_apollo": 32,
            "Parnassius_mnemosyne": 33,
            "Parnassius_glacialis": 34,
            "Sericinus_montela": 35,
            "Zerynthia_rumina": 36,
            "Zerynthia_polyxena": 37,
            "Allancastria_cerisyi": 38,
            "Allancastria_deyrollei": 39,
            "Allancastria_caucasica": 40,
            "Bhutanitis_thaidina": 41,
            "Bhutanitis_lidderdalii": 42,
            "Bhutanitis_mansfieldi": 43,
            "Luehdorfia_japonica": 44,
            "Luehdorfia_puziloi": 45,
            "Luehdorfia_chinensis": 46,
            "Papilio_machaon": 47,
            "Parnassius_stubbendorfii": 48,
            "Parnassius_apollonius": 49,
            "Papilio_alexanor": 50,
            "Papilio_hospiton": 51,
            "Papilio_xuthus": 52,
            "Iphiclides_podalirius": 53,
            "Iphiclides_feisthamelii": 54,
            "Leptidea_sinapis": 55,
            "Colias_palaeno": 56,
            "Colias_pelidne": 57,
            "Leptidea_juvernica": 58,
            "Leptidea_morsei": 59,
            "Leptidea_amurensis": 60,
            "Leptidea_duponcheli": 61,
            "Colias_marcopolo": 62,
            "Colias_ladakensis": 63,
            "Colias_nebulosa": 64,
            "Colias_nastes": 65,
            "Colias_cocandica": 66,
            "Colias_sieversi": 67,
            "Colias_sifanica": 68,
            "Colias_alpherakii": 69,
            "Colias_christophi": 70,
            "Colias_tyche": 71,
            "Colias_phicomone": 72,
            "Colias_alfacariensis": 73,
            "Colias_hyale": 74,
            "Colias_erate": 75,
            "Colias_erschoffi": 76,
            "Colias_romanovi": 77,
            "Colias_regia": 78,
            "Colias_stoliczkana": 79,
            "Colias_hecla": 80,
            "Colias_eogene": 81,
            "Colias_thisoa": 82,
            "Colias_staudingeri": 83,
            "Colias_lada": 84,
            "Colias_baeckeri": 85,
            "Colias_fieldii": 86,
            "Colias_heos": 87,
            "Colias_caucasica": 88,
            "Colias_diva": 89,
            "Colias_chrysotheme": 90,
            "Colias_balcanica": 91,
            "Colias_myrmidone": 92,
            "Colias_croceus": 93,
            "Colias_felderi": 94,
            "Colias_viluiensis": 95,
            "Aporia_crataegi": 96,
            "Colias_aurorina": 97,
            "Colias_chlorocoma": 98,
            "Colias_libanotica": 99,
            "Colias_wiskotti": 100,
            "Catopsilia_florella": 101,
            "Gonepteryx_rhamni": 102,
            "Gonepteryx_maxima": 103,
            "Gonepteryx_cleopatra": 104,
            "Gonepteryx_cleobule": 105,
            "Gonepteryx_amintha": 106,
            "Aporia_procris": 107,
            "Mesapia_peloria": 108,
            "Aporia_potanini": 109,
            "Aporia_nabellica": 110,
            "Baltia_butleri": 111,
            "Pieris_brassicae": 112,
            "Pieris_cheiranthi": 113,
            "Pieris_rapae": 114,
            "Erebia_gorge": 115,
            "Erebia_aethiopellus": 116,
            "Erebia_mnestra": 117,
            "Erebia_epistygne": 118,
            "Erebia_ottomana": 119,
            "Erebia_tyndarus": 120,
            "Erebia_oeme": 121,
            "Erebia_lefebvrei": 122,
            "Erebia_melas": 123,
            "Erebia_zapateri": 124,
            "Erebia_neoridas": 125,
            "Erebia_montana": 126,
            "Erebia_cassioides": 127,
            "Erebia_nivalis": 128,
            "Erebia_scipio": 129,
            "Erebia_pronoe": 130,
            "Erebia_styx": 131,
            "Erebia_meolans": 132,
            "Erebia_palarica": 133,
            "Erebia_pandrose": 134,
            "Erebia_meta": 135,
            "Erebia_erinnyn": 136,
            "Berberia_lambessanus": 137,
            "Berberia_abdelkader": 138,
            "Proterebia_afra": 139,
            "Boeberia_parmenio": 140,
            "Loxerebia_saxicola": 141,
            "Pieris_mannii": 142,
            "Pieris_ergane": 143,
            "Pieris_krueperi": 144,
            "Pieris_napi": 145,
            "Lycaena_thersamon": 146,
            "Lycaena_lampon": 147,
            "Lycaena_solskyi": 148,
            "Lycaena_splendens": 149,
            "Lycaena_candens": 150,
            "Lycaena_ochimus": 151,
            "Lycaena_hippothoe": 152,
            "Lycaena_tityrus": 153,
            "Lycaena_thetis": 154,
            "Melitaea_athalia": 155,
            "Argynnis_paphia": 156,
            "Heliophorus_tamu": 157,
            "Heliophorus_brahma": 158,
            "Heliophorus_androcles": 159,
            "Cethosia_biblis": 160,
            "Childrena_childreni": 161,
            "Melitaea_parthenoides": 162,
            "Pieris_bryoniae": 163,
            "Pontia_edusa": 164,
            "Pontia_daplidice": 165,
            "Pontia_callidice": 166,
            "Anthocharis_thibetana": 167,
            "Anthocharis_bambusarum": 168,
            "Anthocharis_bieti": 169,
            "Anthocharis_scolymus": 170,
            "Zegris_pyrothoe": 171,
            "Zegris_eupheme": 172,
            "Zegris_fausti": 173,
            "Euchloe_simplonia": 174,
            "Pontia_chloridice": 175,
            "Euchloe_belemia": 176,
            "Euchloe_ausonia": 177,
            "Euchloe_tagis": 178,
            "Euchloe_crameri": 179,
            "Euchloe_insularis": 180,
            "Euchloe_orientalis": 181,
            "Euchloe_transcaspica": 182,
            "Euchloe_charlonia": 183,
            "Euchloe_tomyris": 184,
            "Anthocharis_gruneri": 185,
            "Anthocharis_damone": 186,
            "Anthocharis_cardamines": 187,
            "Anthocharis_belia": 188,
            "Anthocharis_euphenoides": 189,
            "Colotis_fausta": 190,
            "Colotis_evagore": 191,
            "Hamearis_lucina": 192,
            "Polycaena_tamerlana": 193,
            "Lycaena_phlaeas": 194,
            "Lycaena_helle": 195,
            "Lycaena_pang": 196,
            "Lycaena_caspius": 197,
            "Lycaena_margelanica": 198,
            "Lycaena_dispar": 199,
            "Lycaena_alciphron": 200,
            "Lycaena_virgaureae": 201,
            "Lycaena_kasyapa": 202,
            "Favonius_quercus": 203,
            "Cigaritis_siphax": 204,
            "Cigaritis_allardi": 205,
            "Tomares_ballus": 206,
            "Tomares_nogelii": 207,
            "Tomares_mauretanicus": 208,
            "Tomares_romanovi": 209,
            "Tomares_callimachus": 210,
            "Chrysozephyrus_smaragdinus": 211,
            "Ussuriana_micahaelis": 212,
            "Coreana_raphaelis": 213,
            "Japonica_saepestriata": 214,
            "Thecla_betulae": 215,
            "Celastrina_argiolus": 216,
            "Laeosopis_roboris": 217,
            "Callophrys_rubi": 218,
            "Zizeeria_knysna": 219,
            "Tarucus_theophrastus": 220,
            "Cyclyrius_webbianus": 221,
            "Tarucus_balkanica": 222,
            "Leptotes_pirithous": 223,
            "Satyrium_spini": 224,
            "Lampides_boeticus": 225,
            "Satyrium_w-album": 226,
            "Satyrium_ilicis": 227,
            "Satyrium_pruni": 228,
            "Satyrium_acaciae": 229,
            "Satyrium_esculi": 230,
            "Neolycaena_rhymnus": 231,
            "Callophrys_avis": 232,
            "Cupido_minimus": 233,
            "Maculinea_rebeli": 234,
            "Maculinea_arion": 235,
            "Cupido_alcetas": 236,
            "Cupido_osiris": 237,
            "Cupido_argiades": 238,
            "Cupido_decolorata": 239,
            "Glaucopsyche_melanops": 240,
            "Glaucopsyche_alexis": 241,
            "Maculinea_alcon": 242,
            "Maculinea_teleius": 243,
            "Pseudophilotes_abencerragus": 244,
            "Pseudophilotes_panoptes": 245,
            "Pseudophilotes_vicrama": 246,
            "Pseudophilotes_baton": 247,
            "Maculinea_nausithous": 248,
            "Scolitantides_orion": 249,
            "Iolana_gigantea": 250,
            "Iolana_iolas": 251,
            "Plebejus_argus": 252,
            "Plebejus_eversmanni": 253,
            "Glaucopsyche_paphos": 254,
            "Plebejus_argyrognomon": 255,
            "Agriades_optilete": 256,
            "Plebejidea_loewii": 257,
            "Plebejus_idas": 258,
            "Kretania_trappi": 259,
            "Kretania_pylaon": 260,
            "Kretania_martini": 261,
            "Plebejus_samudra": 262,
            "Agriades_orbitulus": 263,
            "Aricia_artaxerxes": 264,
            "Pamiria_omphisa": 265,
            "Agriades_glandon": 266,
            "Aricia_agestis": 267,
            "Polyommatus_damon": 268,
            "Eumedonia_eumedon": 269,
            "Aricia_nicias": 270,
            "Cyaniris_semiargus": 271,
            "Polyommatus_dolus": 272,
            "Aricia_anteros": 273,
            "Polyommatus_antidolus": 274,
            "Polyommatus_phyllis": 275,
            "Polyommatus_iphidamon": 276,
            "Polyommatus_damonides": 277,
            "Polyommatus_damone": 278,
            "Polyommatus_ripartii": 279,
            "Polyommatus_admetus": 280,
            "Polyommatus_dorylas": 281,
            "Polyommatus_erschoffi": 282,
            "Polyommatus_thersites": 283,
            "Polyommatus_escheri": 284,
            "Lysandra_bellargus": 285,
            "Lysandra_coridon": 286,
            "Lysandra_hispana": 287,
            "Lysandra_albicans": 288,
            "Lysandra_caelestissima": 289,
            "Lysandra_punctifera": 290,
            "Polyommatus_nivescens": 291,
            "Polyommatus_aedon": 292,
            "Polyommatus_atys": 293,
            "Polyommatus_icarus": 294,
            "Polyommatus_caeruleus": 295,
            "Glabroculus_elvira": 296,
            "Glabroculus_cyane": 297,
            "Polyommatus_stoliczkana": 298,
            "Polyommatus_golgus": 299,
            "Neolysandra_coelestina": 300,
            "Neolysandra_corona": 301,
            "Polyommatus_amandus": 302,
            "Polyommatus_daphnis": 303,
            "Polyommatus_eros": 304,
            "Polyommatus_celina": 305,
            "Libythea_celtis": 306,
            "Danaus_plexippus": 307,
            "Danaus_chrysippus": 308,
            "Charaxes_jasius": 309,
            "Apatura_iris": 310,
            "Apatura_ilia": 311,
            "Limenitis_reducta": 312,
            "Apatura_metis": 313,
            "Euapatura_mirza": 314,
            "Hestina_japonica": 315,
            "Timelaea_albescens": 316,
            "Limenitis_populi": 317,
            "Limenitis_camilla": 318,
            "Mimathyma_schrenckii": 319,
            "Limenitis_sydyi": 320,
            "Lelecella_limenitoides": 321,
            "Neptis_sappho": 322,
            "Neptis_rivularis": 323,
            "Nymphalis_antiopa": 324,
            "Nymphalis_polychloros": 325,
            "Nymphalis_xanthomelas": 326,
            "Nymphalis_l-album": 327,
            "Nymphalis_urticae": 328,
            "Nymphalis_ichnusa": 329,
            "Nymphalis_egea": 330,
            "Nymphalis_c-album": 331,
            "Inachis_io": 332,
            "Araschnia_burejana": 333,
            "Araschnia_levana": 334,
            "Nymphalis_canace": 335,
            "Nymphalis_c-aureum": 336,
            "Vanessa_atalanta": 337,
            "Vanessa_vulcania": 338,
            "Vanessa_cardui": 339,
            "Argynnis_pandora": 340,
            "Speyeria_aglaja": 341,
            "Fabriciana_niobe": 342,
            "Speyeria_clara": 343,
            "Argyronome_laodice": 344,
            "Fabriciana_adippe": 345,
            "Fabriciana_jainadeva": 346,
            "Fabriciana_auresiana": 347,
            "Fabriciana_elisa": 348,
            "Issoria_lathonia": 349,
            "Brenthis_hecate": 350,
            "Brenthis_daphne": 351,
            "Brenthis_ino": 352,
            "Boloria_pales": 353,
            "Kuekenthaliella_eugenia": 354,
            "Boloria_aquilonaris": 355,
            "Boloria_napaea": 356,
            "Clossiana_selene": 357,
            "Proclossiana_eunomia": 358,
            "Boloria_graeca": 359,
            "Clossiana_thore": 360,
            "Clossiana_dia": 361,
            "Clossiana_euphrosyne": 362,
            "Clossiana_titania": 363,
            "Clossiana_freija": 364,
            "Melitaea_cinxia": 365,
            "Melitaea_phoebe": 366,
            "Melitaea_didyma": 367,
            "Melitaea_varia": 368,
            "Melitaea_aurelia": 369,
            "Melitaea_asteria": 370,
            "Melitaea_diamina": 371,
            "Melitaea_britomartis": 372,
            "Melitaea_acraeina": 373,
            "Melitaea_trivia": 374,
            "Melitaea_persea": 375,
            "Melitaea_ambigua": 376,
            "Melitaea_deione": 377,
            "Melitaea_turanica": 378,
            "Euphydryas_maturna": 379,
            "Euphydryas_ichnea": 380,
            "Euphydryas_cynthia": 381,
            "Euphydryas_aurinia": 382,
            "Euphydryas_sibirica": 383,
            "Euphydryas_iduna": 384,
            "Melanargia_titea": 385,
            "Melanargia_parce": 386,
            "Melanargia_lachesis": 387,
            "Melanargia_galathea": 388,
            "Melanargia_russiae": 389,
            "Melanargia_larissa": 390,
            "Melanargia_ines": 391,
            "Melanargia_pherusa": 392,
            "Melanargia_occitanica": 393,
            "Melanargia_arge": 394,
            "Melanargia_meridionalis": 395,
            "Melanargia_leda": 396,
            "Melanargia_halimede": 397,
            "Davidina_armandi": 398,
            "Hipparchia_semele": 399,
            "Chazara_briseis": 400,
            "Hipparchia_parisatis": 401,
            "Hipparchia_fidia": 402,
            "Hipparchia_genava": 403,
            "Hipparchia_aristaeus": 404,
            "Hipparchia_fagi": 405,
            "Hipparchia_wyssii": 406,
            "Hipparchia_fatua": 407,
            "Hipparchia_statilinus": 408,
            "Hipparchia_syriaca": 409,
            "Hipparchia_neomiris": 410,
            "Hipparchia_azorina": 411,
            "Chazara_prieuri": 412,
            "Chazara_bischoffii": 413,
            "Chazara_persephone": 414,
            "Pseudochazara_pelopea": 415,
            "Pseudochazara_beroe": 416,
            "Pseudochazara_schahrudensis": 417,
            "Pseudochazara_telephassa": 418,
            "Pseudochazara_anthelea": 419,
            "Pseudochazara_amalthea": 420,
            "Pseudochazara_graeca": 421,
            "Pseudochazara_cingovskii": 422,
            "Karanasa_modesta": 423,
            "Oeneis_magna": 424,
            "Oeneis_glacialis": 425,
            "Satyrus_actaea": 426,
            "Satyrus_parthicus": 427,
            "Satyrus_ferula": 428,
            "Minois_dryas": 429,
            "Arethusana_arethusa": 430,
            "Brintesia_circe": 431,
            "Maniola_jurtina": 432,
            "Aphantopus_hyperantus": 433,
            "Hyponephele_pulchra": 434,
            "Hyponephele_pulchella": 435,
            "Hyponephele_cadusia": 436,
            "Hyponephele_amardaea": 437,
            "Hyponephele_lycaon": 438,
            "Maniola_nurag": 439,
            "Hyponephele_lupina": 440,
            "Pyronia_tithonus": 441,
            "Coenonympha_gardetta": 442,
            "Coenonympha_tullia": 443,
            "Pyronia_bathseba": 444,
            "Pyronia_cecilia": 445,
            "Coenonympha_corinna": 446,
            "Coenonympha_pamphilus": 447,
            "Pyronia_janiroides": 448,
            "Coenonympha_dorus": 449,
            "Coenonympha_darwiniana": 450,
            "Coenonympha_arcania": 451,
            "Pararge_aegeria": 452,
            "Coenonympha_leander": 453,
            "Ypthima_baldus": 454,
            "Coenonympha_iphioides": 455,
            "Coenonympha_glycerion": 456,
            "Coenonympha_hero": 457,
            "Coenonympha_oedippus": 458,
            "Pararge_xiphioides": 459,
            "Lasiommata_megera": 460,
            "Lasiommata_petropolitana": 461,
            "Lasiommata_maera": 462,
            "Lasiommata_paramegaera": 463,
            "Lopinga_achine": 464,
            "Erebia_euryale": 465,
            "Kirinia_roxelana": 466,
            "Kirinia_climene": 467,
            "Neope_goschkevitschii": 468,
            "Erebia_ligea": 469,
            "Kirinia_eversmanni": 470,
            "Erebia_eriphyle": 471,
            "Erebia_manto": 472,
            "Erebia_epiphron": 473,
            "Erebia_flavofasciata": 474,
            "Erebia_bubastis": 475,
            "Erebia_claudina": 476,
            "Erebia_christi": 477,
            "Erebia_pharte": 478,
            "Erebia_aethiops": 479,
            "Erebia_melampus": 480,
            "Erebia_sudetica": 481,
            "Erebia_neriene": 482,
            "Erebia_triaria": 483,
            "Erebia_medusa": 484,
            "Erebia_alberganus": 485,
            "Erebia_pluto": 486,
            "Gonepteryx_farinosa": 487,
            "Melitaea_nevadensis": 488,
            "Agriades_pheretiades": 489,
            "Parnassius_eversmannii": 490,
            "Parnassius_ariadne": 491,
            "Parnassius_stenosemus": 492,
            "Parnassius_hardwickii": 493,
            "Parnassius_charltonius": 494,
            "Parnassius_imperator": 495,
            "Parnassius_acdestis": 496,
            "Parnassius_cardinal": 497,
            "Parnassius_szechenyii": 498,
            "Parnassius_delphius": 499,
            "Parnassius_maximinus": 500,
            "Parnassius_staudingeri": 501,
            "Parnassius_orleans": 502,
            "Parnassius_augustus": 503,
            "Parnassius_loxias": 504,
            "Parnassius_charltontonius": 505,
            "Parnassius_autocrator": 506,
            "Parnassius_stoliczkanus": 507,
            "Parnassius_nordmanni": 508,
            "Parnassius_simo": 509,
            "Parnassius_bremeri": 510,
            "Parnassius_actius": 511,
            "Parnassius_cephalus": 512,
            "Parnassius_maharaja": 513,
            "Parnassius_tenedius": 514,
            "Parnassius_acco": 515,
            "Parnassius_boedromius": 516,
            "Parnassius_tianschanicus": 517,
            "Parnassius_phoebus": 518,
            "Parnassius_honrathi": 519,
            "Parnassius_ruckbeili": 520,
            "Parnassius_epaphus": 521,
            "Parnassius_nomion": 522,
            "Parnassius_jacquemonti": 523,
            "Parnassius_mercurius": 524,
            "Parnassius_tibetanus": 525,
            "Parnassius_clodius": 526,
            "Parnassius_smintheus": 527,
            "Parnassius_behrii": 528,
            "Atrophaneura_mencius": 529,
            "Atrophaneura_plutonius": 530,
            "Papilio_dehaani": 531,
            "Papilio_polytes": 532,
            "Atrophaneura_horishana": 533,
            "Papilio_bootes": 534,
            "Agehana_elwesi": 535,
            "Papilio_maackii": 536,
            "Atrophaneura_impediens": 537,
            "Atrophaneura_polyeuctes": 538,
            "Arisbe_mandarinus": 539,
            "Arisbe_parus": 540,
            "Atrophaneura_alcinous": 541,
            "Arisbe_alebion": 542,
            "Papilio_helenus": 543,
            "Teinopalpus_imperialis": 544,
            "Arisbe_eurous": 545,
            "Graphium_sarpedon": 546,
            "Arisbe_doson": 547,
            "Arisbe_tamerlanus": 548,
            "Papilio_bianor": 549,
            "Papilio_paris": 550,
            "Atrophaneura_nevilli": 551,
            "Papilio_krishna": 552,
            "Papilio_macilentus": 553,
            "Arisbe_leechi": 554,
            "Papilio_protenor": 555,
            "Graphium_cloanthus": 556,
            "Papilio_castor": 557,
            "Meandrusa_sciron": 558,
            "Papilio_arcturus": 559,
            "Agriades_lehanus": 560
        }

        self.child_of_family = {
            "Hesperiidae": [
                "Heteropterinae",
                "Hesperiinae",
                "Pyrginae"
            ],
            "Papilionidae": [
                "Parnassiinae",
                "Papilioninae"
            ],
            "Pieridae": [
                "Dismorphiinae",
                "Coliadinae",
                "Pierinae"
            ],
            "Nymphalidae": [
                "Satyrinae",
                "Nymphalinae",
                "Heliconiinae",
                "Libytheinae",
                "Danainae",
                "Charaxinae",
                "Apaturinae",
                "Limenitidinae"
            ],
            "Lycaenidae": [
                "Lycaeninae",
                "Theclinae",
                "Aphnaeinae",
                "Polyommatinae"
            ],
            "Riodinidae": [
                "Nemeobiinae"
            ]
        }

        self.child_of_subfamily = {
            "Heteropterinae": [
                "Carterocephalus",
                "Heteropterus"
            ],
            "Hesperiinae": [
                "Thymelicus",
                "Hesperia",
                "Ochlodes",
                "Gegenes"
            ],
            "Pyrginae": [
                "Erynnis",
                "Carcharodus",
                "Spialia",
                "Muschampia",
                "Pyrgus"
            ],
            "Parnassiinae": [
                "Parnassius",
                "Archon",
                "Sericinus",
                "Zerynthia",
                "Allancastria",
                "Bhutanitis",
                "Luehdorfia"
            ],
            "Papilioninae": [
                "Papilio",
                "Iphiclides",
                "Atrophaneura",
                "Agehana",
                "Arisbe",
                "Teinopalpus",
                "Graphium",
                "Meandrusa"
            ],
            "Dismorphiinae": [
                "Leptidea"
            ],
            "Coliadinae": [
                "Colias",
                "Catopsilia",
                "Gonepteryx"
            ],
            "Pierinae": [
                "Aporia",
                "Mesapia",
                "Baltia",
                "Pieris",
                "Pontia",
                "Anthocharis",
                "Zegris",
                "Euchloe",
                "Colotis"
            ],
            "Satyrinae": [
                "Erebia",
                "Berberia",
                "Proterebia",
                "Boeberia",
                "Loxerebia",
                "Melanargia",
                "Davidina",
                "Hipparchia",
                "Chazara",
                "Pseudochazara",
                "Karanasa",
                "Oeneis",
                "Satyrus",
                "Minois",
                "Arethusana",
                "Brintesia",
                "Maniola",
                "Aphantopus",
                "Hyponephele",
                "Pyronia",
                "Coenonympha",
                "Pararge",
                "Ypthima",
                "Lasiommata",
                "Lopinga",
                "Kirinia",
                "Neope"
            ],
            "Lycaeninae": [
                "Lycaena",
                "Heliophorus"
            ],
            "Nymphalinae": [
                "Melitaea",
                "Nymphalis",
                "Inachis",
                "Araschnia",
                "Vanessa",
                "Euphydryas"
            ],
            "Heliconiinae": [
                "Argynnis",
                "Cethosia",
                "Childrena",
                "Speyeria",
                "Fabriciana",
                "Argyronome",
                "Issoria",
                "Brenthis",
                "Boloria",
                "Kuekenthaliella",
                "Clossiana",
                "Proclossiana"
            ],
            "Nemeobiinae": [
                "Hamearis",
                "Polycaena"
            ],
            "Theclinae": [
                "Favonius",
                "Tomares",
                "Chrysozephyrus",
                "Ussuriana",
                "Coreana",
                "Japonica",
                "Thecla",
                "Laeosopis",
                "Callophrys",
                "Satyrium",
                "Neolycaena"
            ],
            "Aphnaeinae": [
                "Cigaritis"
            ],
            "Polyommatinae": [
                "Celastrina",
                "Zizeeria",
                "Tarucus",
                "Cyclyrius",
                "Leptotes",
                "Lampides",
                "Cupido",
                "Maculinea",
                "Glaucopsyche",
                "Pseudophilotes",
                "Scolitantides",
                "Iolana",
                "Plebejus",
                "Agriades",
                "Plebejidea",
                "Kretania",
                "Aricia",
                "Pamiria",
                "Polyommatus",
                "Eumedonia",
                "Cyaniris",
                "Lysandra",
                "Glabroculus",
                "Neolysandra"
            ],
            "Libytheinae": [
                "Libythea"
            ],
            "Danainae": [
                "Danaus"
            ],
            "Charaxinae": [
                "Charaxes"
            ],
            "Apaturinae": [
                "Apatura",
                "Euapatura",
                "Hestina",
                "Timelaea",
                "Mimathyma"
            ],
            "Limenitidinae": [
                "Limenitis",
                "Lelecella",
                "Neptis"
            ]
        }

        self.child_of_genus = {
            "Carterocephalus": [
                "Carterocephalus_palaemon"
            ],
            "Heteropterus": [
                "Heteropterus_morpheus"
            ],
            "Thymelicus": [
                "Thymelicus_sylvestris",
                "Thymelicus_lineola",
                "Thymelicus_acteon"
            ],
            "Hesperia": [
                "Hesperia_comma"
            ],
            "Ochlodes": [
                "Ochlodes_venata"
            ],
            "Gegenes": [
                "Gegenes_nostrodamus"
            ],
            "Erynnis": [
                "Erynnis_tages"
            ],
            "Carcharodus": [
                "Carcharodus_alceae",
                "Carcharodus_lavatherae",
                "Carcharodus_baeticus",
                "Carcharodus_floccifera"
            ],
            "Spialia": [
                "Spialia_sertorius",
                "Spialia_orbifer"
            ],
            "Muschampia": [
                "Muschampia_proto"
            ],
            "Pyrgus": [
                "Pyrgus_alveus",
                "Pyrgus_armoricanus",
                "Pyrgus_andromedae",
                "Pyrgus_cacaliae",
                "Pyrgus_carlinae",
                "Pyrgus_carthami",
                "Pyrgus_malvae",
                "Pyrgus_cinarae",
                "Pyrgus_cirsii",
                "Pyrgus_malvoides",
                "Pyrgus_onopordi",
                "Pyrgus_serratulae",
                "Pyrgus_sidae",
                "Pyrgus_warrenensis"
            ],
            "Parnassius": [
                "Parnassius_sacerdos",
                "Parnassius_apollo",
                "Parnassius_mnemosyne",
                "Parnassius_glacialis",
                "Parnassius_stubbendorfii",
                "Parnassius_apollonius",
                "Parnassius_eversmannii",
                "Parnassius_ariadne",
                "Parnassius_stenosemus",
                "Parnassius_hardwickii",
                "Parnassius_charltonius",
                "Parnassius_imperator",
                "Parnassius_acdestis",
                "Parnassius_cardinal",
                "Parnassius_szechenyii",
                "Parnassius_delphius",
                "Parnassius_maximinus",
                "Parnassius_staudingeri",
                "Parnassius_orleans",
                "Parnassius_augustus",
                "Parnassius_loxias",
                "Parnassius_charltontonius",
                "Parnassius_autocrator",
                "Parnassius_stoliczkanus",
                "Parnassius_nordmanni",
                "Parnassius_simo",
                "Parnassius_bremeri",
                "Parnassius_actius",
                "Parnassius_cephalus",
                "Parnassius_maharaja",
                "Parnassius_tenedius",
                "Parnassius_acco",
                "Parnassius_boedromius",
                "Parnassius_tianschanicus",
                "Parnassius_phoebus",
                "Parnassius_honrathi",
                "Parnassius_ruckbeili",
                "Parnassius_epaphus",
                "Parnassius_nomion",
                "Parnassius_jacquemonti",
                "Parnassius_mercurius",
                "Parnassius_tibetanus",
                "Parnassius_clodius",
                "Parnassius_smintheus",
                "Parnassius_behrii"
            ],
            "Archon": [
                "Archon_apollinus"
            ],
            "Sericinus": [
                "Sericinus_montela"
            ],
            "Zerynthia": [
                "Zerynthia_rumina",
                "Zerynthia_polyxena"
            ],
            "Allancastria": [
                "Allancastria_cerisyi",
                "Allancastria_deyrollei",
                "Allancastria_caucasica"
            ],
            "Bhutanitis": [
                "Bhutanitis_thaidina",
                "Bhutanitis_lidderdalii",
                "Bhutanitis_mansfieldi"
            ],
            "Luehdorfia": [
                "Luehdorfia_japonica",
                "Luehdorfia_puziloi",
                "Luehdorfia_chinensis"
            ],
            "Papilio": [
                "Papilio_machaon",
                "Papilio_alexanor",
                "Papilio_hospiton",
                "Papilio_xuthus",
                "Papilio_dehaani",
                "Papilio_polytes",
                "Papilio_bootes",
                "Papilio_maackii",
                "Papilio_helenus",
                "Papilio_bianor",
                "Papilio_paris",
                "Papilio_krishna",
                "Papilio_macilentus",
                "Papilio_protenor",
                "Papilio_castor",
                "Papilio_arcturus"
            ],
            "Iphiclides": [
                "Iphiclides_podalirius",
                "Iphiclides_feisthamelii"
            ],
            "Leptidea": [
                "Leptidea_sinapis",
                "Leptidea_juvernica",
                "Leptidea_morsei",
                "Leptidea_amurensis",
                "Leptidea_duponcheli"
            ],
            "Colias": [
                "Colias_palaeno",
                "Colias_pelidne",
                "Colias_marcopolo",
                "Colias_ladakensis",
                "Colias_nebulosa",
                "Colias_nastes",
                "Colias_cocandica",
                "Colias_sieversi",
                "Colias_sifanica",
                "Colias_alpherakii",
                "Colias_christophi",
                "Colias_tyche",
                "Colias_phicomone",
                "Colias_alfacariensis",
                "Colias_hyale",
                "Colias_erate",
                "Colias_erschoffi",
                "Colias_romanovi",
                "Colias_regia",
                "Colias_stoliczkana",
                "Colias_hecla",
                "Colias_eogene",
                "Colias_thisoa",
                "Colias_staudingeri",
                "Colias_lada",
                "Colias_baeckeri",
                "Colias_fieldii",
                "Colias_heos",
                "Colias_caucasica",
                "Colias_diva",
                "Colias_chrysotheme",
                "Colias_balcanica",
                "Colias_myrmidone",
                "Colias_croceus",
                "Colias_felderi",
                "Colias_viluiensis",
                "Colias_aurorina",
                "Colias_chlorocoma",
                "Colias_libanotica",
                "Colias_wiskotti"
            ],
            "Aporia": [
                "Aporia_crataegi",
                "Aporia_procris",
                "Aporia_potanini",
                "Aporia_nabellica"
            ],
            "Catopsilia": [
                "Catopsilia_florella"
            ],
            "Gonepteryx": [
                "Gonepteryx_rhamni",
                "Gonepteryx_maxima",
                "Gonepteryx_cleopatra",
                "Gonepteryx_cleobule",
                "Gonepteryx_amintha",
                "Gonepteryx_farinosa"
            ],
            "Mesapia": [
                "Mesapia_peloria"
            ],
            "Baltia": [
                "Baltia_butleri"
            ],
            "Pieris": [
                "Pieris_brassicae",
                "Pieris_cheiranthi",
                "Pieris_rapae",
                "Pieris_mannii",
                "Pieris_ergane",
                "Pieris_krueperi",
                "Pieris_napi",
                "Pieris_bryoniae"
            ],
            "Erebia": [
                "Erebia_gorge",
                "Erebia_aethiopellus",
                "Erebia_mnestra",
                "Erebia_epistygne",
                "Erebia_ottomana",
                "Erebia_tyndarus",
                "Erebia_oeme",
                "Erebia_lefebvrei",
                "Erebia_melas",
                "Erebia_zapateri",
                "Erebia_neoridas",
                "Erebia_montana",
                "Erebia_cassioides",
                "Erebia_nivalis",
                "Erebia_scipio",
                "Erebia_pronoe",
                "Erebia_styx",
                "Erebia_meolans",
                "Erebia_palarica",
                "Erebia_pandrose",
                "Erebia_meta",
                "Erebia_erinnyn",
                "Erebia_euryale",
                "Erebia_ligea",
                "Erebia_eriphyle",
                "Erebia_manto",
                "Erebia_epiphron",
                "Erebia_flavofasciata",
                "Erebia_bubastis",
                "Erebia_claudina",
                "Erebia_christi",
                "Erebia_pharte",
                "Erebia_aethiops",
                "Erebia_melampus",
                "Erebia_sudetica",
                "Erebia_neriene",
                "Erebia_triaria",
                "Erebia_medusa",
                "Erebia_alberganus",
                "Erebia_pluto"
            ],
            "Berberia": [
                "Berberia_lambessanus",
                "Berberia_abdelkader"
            ],
            "Proterebia": [
                "Proterebia_afra"
            ],
            "Boeberia": [
                "Boeberia_parmenio"
            ],
            "Loxerebia": [
                "Loxerebia_saxicola"
            ],
            "Lycaena": [
                "Lycaena_thersamon",
                "Lycaena_lampon",
                "Lycaena_solskyi",
                "Lycaena_splendens",
                "Lycaena_candens",
                "Lycaena_ochimus",
                "Lycaena_hippothoe",
                "Lycaena_tityrus",
                "Lycaena_thetis",
                "Lycaena_phlaeas",
                "Lycaena_helle",
                "Lycaena_pang",
                "Lycaena_caspius",
                "Lycaena_margelanica",
                "Lycaena_dispar",
                "Lycaena_alciphron",
                "Lycaena_virgaureae",
                "Lycaena_kasyapa"
            ],
            "Melitaea": [
                "Melitaea_athalia",
                "Melitaea_parthenoides",
                "Melitaea_cinxia",
                "Melitaea_phoebe",
                "Melitaea_didyma",
                "Melitaea_varia",
                "Melitaea_aurelia",
                "Melitaea_asteria",
                "Melitaea_diamina",
                "Melitaea_britomartis",
                "Melitaea_acraeina",
                "Melitaea_trivia",
                "Melitaea_persea",
                "Melitaea_ambigua",
                "Melitaea_deione",
                "Melitaea_turanica",
                "Melitaea_nevadensis"
            ],
            "Argynnis": [
                "Argynnis_paphia",
                "Argynnis_pandora"
            ],
            "Heliophorus": [
                "Heliophorus_tamu",
                "Heliophorus_brahma",
                "Heliophorus_androcles"
            ],
            "Cethosia": [
                "Cethosia_biblis"
            ],
            "Childrena": [
                "Childrena_childreni"
            ],
            "Pontia": [
                "Pontia_edusa",
                "Pontia_daplidice",
                "Pontia_callidice",
                "Pontia_chloridice"
            ],
            "Anthocharis": [
                "Anthocharis_thibetana",
                "Anthocharis_bambusarum",
                "Anthocharis_bieti",
                "Anthocharis_scolymus",
                "Anthocharis_gruneri",
                "Anthocharis_damone",
                "Anthocharis_cardamines",
                "Anthocharis_belia",
                "Anthocharis_euphenoides"
            ],
            "Zegris": [
                "Zegris_pyrothoe",
                "Zegris_eupheme",
                "Zegris_fausti"
            ],
            "Euchloe": [
                "Euchloe_simplonia",
                "Euchloe_belemia",
                "Euchloe_ausonia",
                "Euchloe_tagis",
                "Euchloe_crameri",
                "Euchloe_insularis",
                "Euchloe_orientalis",
                "Euchloe_transcaspica",
                "Euchloe_charlonia",
                "Euchloe_tomyris"
            ],
            "Colotis": [
                "Colotis_fausta",
                "Colotis_evagore"
            ],
            "Hamearis": [
                "Hamearis_lucina"
            ],
            "Polycaena": [
                "Polycaena_tamerlana"
            ],
            "Favonius": [
                "Favonius_quercus"
            ],
            "Cigaritis": [
                "Cigaritis_siphax",
                "Cigaritis_allardi"
            ],
            "Tomares": [
                "Tomares_ballus",
                "Tomares_nogelii",
                "Tomares_mauretanicus",
                "Tomares_romanovi",
                "Tomares_callimachus"
            ],
            "Chrysozephyrus": [
                "Chrysozephyrus_smaragdinus"
            ],
            "Ussuriana": [
                "Ussuriana_micahaelis"
            ],
            "Coreana": [
                "Coreana_raphaelis"
            ],
            "Japonica": [
                "Japonica_saepestriata"
            ],
            "Thecla": [
                "Thecla_betulae"
            ],
            "Celastrina": [
                "Celastrina_argiolus"
            ],
            "Laeosopis": [
                "Laeosopis_roboris"
            ],
            "Callophrys": [
                "Callophrys_rubi",
                "Callophrys_avis"
            ],
            "Zizeeria": [
                "Zizeeria_knysna"
            ],
            "Tarucus": [
                "Tarucus_theophrastus",
                "Tarucus_balkanica"
            ],
            "Cyclyrius": [
                "Cyclyrius_webbianus"
            ],
            "Leptotes": [
                "Leptotes_pirithous"
            ],
            "Satyrium": [
                "Satyrium_spini",
                "Satyrium_w-album",
                "Satyrium_ilicis",
                "Satyrium_pruni",
                "Satyrium_acaciae",
                "Satyrium_esculi"
            ],
            "Lampides": [
                "Lampides_boeticus"
            ],
            "Neolycaena": [
                "Neolycaena_rhymnus"
            ],
            "Cupido": [
                "Cupido_minimus",
                "Cupido_alcetas",
                "Cupido_osiris",
                "Cupido_argiades",
                "Cupido_decolorata"
            ],
            "Maculinea": [
                "Maculinea_rebeli",
                "Maculinea_arion",
                "Maculinea_alcon",
                "Maculinea_teleius",
                "Maculinea_nausithous"
            ],
            "Glaucopsyche": [
                "Glaucopsyche_melanops",
                "Glaucopsyche_alexis",
                "Glaucopsyche_paphos"
            ],
            "Pseudophilotes": [
                "Pseudophilotes_abencerragus",
                "Pseudophilotes_panoptes",
                "Pseudophilotes_vicrama",
                "Pseudophilotes_baton"
            ],
            "Scolitantides": [
                "Scolitantides_orion"
            ],
            "Iolana": [
                "Iolana_gigantea",
                "Iolana_iolas"
            ],
            "Plebejus": [
                "Plebejus_argus",
                "Plebejus_eversmanni",
                "Plebejus_argyrognomon",
                "Plebejus_idas",
                "Plebejus_samudra"
            ],
            "Agriades": [
                "Agriades_optilete",
                "Agriades_orbitulus",
                "Agriades_glandon",
                "Agriades_pheretiades",
                "Agriades_lehanus"
            ],
            "Plebejidea": [
                "Plebejidea_loewii"
            ],
            "Kretania": [
                "Kretania_trappi",
                "Kretania_pylaon",
                "Kretania_martini"
            ],
            "Aricia": [
                "Aricia_artaxerxes",
                "Aricia_agestis",
                "Aricia_nicias",
                "Aricia_anteros"
            ],
            "Pamiria": [
                "Pamiria_omphisa"
            ],
            "Polyommatus": [
                "Polyommatus_damon",
                "Polyommatus_dolus",
                "Polyommatus_antidolus",
                "Polyommatus_phyllis",
                "Polyommatus_iphidamon",
                "Polyommatus_damonides",
                "Polyommatus_damone",
                "Polyommatus_ripartii",
                "Polyommatus_admetus",
                "Polyommatus_dorylas",
                "Polyommatus_erschoffi",
                "Polyommatus_thersites",
                "Polyommatus_escheri",
                "Polyommatus_nivescens",
                "Polyommatus_aedon",
                "Polyommatus_atys",
                "Polyommatus_icarus",
                "Polyommatus_caeruleus",
                "Polyommatus_stoliczkana",
                "Polyommatus_golgus",
                "Polyommatus_amandus",
                "Polyommatus_daphnis",
                "Polyommatus_eros",
                "Polyommatus_celina"
            ],
            "Eumedonia": [
                "Eumedonia_eumedon"
            ],
            "Cyaniris": [
                "Cyaniris_semiargus"
            ],
            "Lysandra": [
                "Lysandra_bellargus",
                "Lysandra_coridon",
                "Lysandra_hispana",
                "Lysandra_albicans",
                "Lysandra_caelestissima",
                "Lysandra_punctifera"
            ],
            "Glabroculus": [
                "Glabroculus_elvira",
                "Glabroculus_cyane"
            ],
            "Neolysandra": [
                "Neolysandra_coelestina",
                "Neolysandra_corona"
            ],
            "Libythea": [
                "Libythea_celtis"
            ],
            "Danaus": [
                "Danaus_plexippus",
                "Danaus_chrysippus"
            ],
            "Charaxes": [
                "Charaxes_jasius"
            ],
            "Apatura": [
                "Apatura_iris",
                "Apatura_ilia",
                "Apatura_metis"
            ],
            "Limenitis": [
                "Limenitis_reducta",
                "Limenitis_populi",
                "Limenitis_camilla",
                "Limenitis_sydyi"
            ],
            "Euapatura": [
                "Euapatura_mirza"
            ],
            "Hestina": [
                "Hestina_japonica"
            ],
            "Timelaea": [
                "Timelaea_albescens"
            ],
            "Mimathyma": [
                "Mimathyma_schrenckii"
            ],
            "Lelecella": [
                "Lelecella_limenitoides"
            ],
            "Neptis": [
                "Neptis_sappho",
                "Neptis_rivularis"
            ],
            "Nymphalis": [
                "Nymphalis_antiopa",
                "Nymphalis_polychloros",
                "Nymphalis_xanthomelas",
                "Nymphalis_l-album",
                "Nymphalis_urticae",
                "Nymphalis_ichnusa",
                "Nymphalis_egea",
                "Nymphalis_c-album",
                "Nymphalis_canace",
                "Nymphalis_c-aureum"
            ],
            "Inachis": [
                "Inachis_io"
            ],
            "Araschnia": [
                "Araschnia_burejana",
                "Araschnia_levana"
            ],
            "Vanessa": [
                "Vanessa_atalanta",
                "Vanessa_vulcania",
                "Vanessa_cardui"
            ],
            "Speyeria": [
                "Speyeria_aglaja",
                "Speyeria_clara"
            ],
            "Fabriciana": [
                "Fabriciana_niobe",
                "Fabriciana_adippe",
                "Fabriciana_jainadeva",
                "Fabriciana_auresiana",
                "Fabriciana_elisa"
            ],
            "Argyronome": [
                "Argyronome_laodice"
            ],
            "Issoria": [
                "Issoria_lathonia"
            ],
            "Brenthis": [
                "Brenthis_hecate",
                "Brenthis_daphne",
                "Brenthis_ino"
            ],
            "Boloria": [
                "Boloria_pales",
                "Boloria_aquilonaris",
                "Boloria_napaea",
                "Boloria_graeca"
            ],
            "Kuekenthaliella": [
                "Kuekenthaliella_eugenia"
            ],
            "Clossiana": [
                "Clossiana_selene",
                "Clossiana_thore",
                "Clossiana_dia",
                "Clossiana_euphrosyne",
                "Clossiana_titania",
                "Clossiana_freija"
            ],
            "Proclossiana": [
                "Proclossiana_eunomia"
            ],
            "Euphydryas": [
                "Euphydryas_maturna",
                "Euphydryas_ichnea",
                "Euphydryas_cynthia",
                "Euphydryas_aurinia",
                "Euphydryas_sibirica",
                "Euphydryas_iduna"
            ],
            "Melanargia": [
                "Melanargia_titea",
                "Melanargia_parce",
                "Melanargia_lachesis",
                "Melanargia_galathea",
                "Melanargia_russiae",
                "Melanargia_larissa",
                "Melanargia_ines",
                "Melanargia_pherusa",
                "Melanargia_occitanica",
                "Melanargia_arge",
                "Melanargia_meridionalis",
                "Melanargia_leda",
                "Melanargia_halimede"
            ],
            "Davidina": [
                "Davidina_armandi"
            ],
            "Hipparchia": [
                "Hipparchia_semele",
                "Hipparchia_parisatis",
                "Hipparchia_fidia",
                "Hipparchia_genava",
                "Hipparchia_aristaeus",
                "Hipparchia_fagi",
                "Hipparchia_wyssii",
                "Hipparchia_fatua",
                "Hipparchia_statilinus",
                "Hipparchia_syriaca",
                "Hipparchia_neomiris",
                "Hipparchia_azorina"
            ],
            "Chazara": [
                "Chazara_briseis",
                "Chazara_prieuri",
                "Chazara_bischoffii",
                "Chazara_persephone"
            ],
            "Pseudochazara": [
                "Pseudochazara_pelopea",
                "Pseudochazara_beroe",
                "Pseudochazara_schahrudensis",
                "Pseudochazara_telephassa",
                "Pseudochazara_anthelea",
                "Pseudochazara_amalthea",
                "Pseudochazara_graeca",
                "Pseudochazara_cingovskii"
            ],
            "Karanasa": [
                "Karanasa_modesta"
            ],
            "Oeneis": [
                "Oeneis_magna",
                "Oeneis_glacialis"
            ],
            "Satyrus": [
                "Satyrus_actaea",
                "Satyrus_parthicus",
                "Satyrus_ferula"
            ],
            "Minois": [
                "Minois_dryas"
            ],
            "Arethusana": [
                "Arethusana_arethusa"
            ],
            "Brintesia": [
                "Brintesia_circe"
            ],
            "Maniola": [
                "Maniola_jurtina",
                "Maniola_nurag"
            ],
            "Aphantopus": [
                "Aphantopus_hyperantus"
            ],
            "Hyponephele": [
                "Hyponephele_pulchra",
                "Hyponephele_pulchella",
                "Hyponephele_cadusia",
                "Hyponephele_amardaea",
                "Hyponephele_lycaon",
                "Hyponephele_lupina"
            ],
            "Pyronia": [
                "Pyronia_tithonus",
                "Pyronia_bathseba",
                "Pyronia_cecilia",
                "Pyronia_janiroides"
            ],
            "Coenonympha": [
                "Coenonympha_gardetta",
                "Coenonympha_tullia",
                "Coenonympha_corinna",
                "Coenonympha_pamphilus",
                "Coenonympha_dorus",
                "Coenonympha_darwiniana",
                "Coenonympha_arcania",
                "Coenonympha_leander",
                "Coenonympha_iphioides",
                "Coenonympha_glycerion",
                "Coenonympha_hero",
                "Coenonympha_oedippus"
            ],
            "Pararge": [
                "Pararge_aegeria",
                "Pararge_xiphioides"
            ],
            "Ypthima": [
                "Ypthima_baldus"
            ],
            "Lasiommata": [
                "Lasiommata_megera",
                "Lasiommata_petropolitana",
                "Lasiommata_maera",
                "Lasiommata_paramegaera"
            ],
            "Lopinga": [
                "Lopinga_achine"
            ],
            "Kirinia": [
                "Kirinia_roxelana",
                "Kirinia_climene",
                "Kirinia_eversmanni"
            ],
            "Neope": [
                "Neope_goschkevitschii"
            ],
            "Atrophaneura": [
                "Atrophaneura_mencius",
                "Atrophaneura_plutonius",
                "Atrophaneura_horishana",
                "Atrophaneura_impediens",
                "Atrophaneura_polyeuctes",
                "Atrophaneura_alcinous",
                "Atrophaneura_nevilli"
            ],
            "Agehana": [
                "Agehana_elwesi"
            ],
            "Arisbe": [
                "Arisbe_mandarinus",
                "Arisbe_parus",
                "Arisbe_alebion",
                "Arisbe_eurous",
                "Arisbe_doson",
                "Arisbe_tamerlanus",
                "Arisbe_leechi"
            ],
            "Teinopalpus": [
                "Teinopalpus_imperialis"
            ],
            "Graphium": [
                "Graphium_sarpedon",
                "Graphium_cloanthus"
            ],
            "Meandrusa": [
                "Meandrusa_sciron"
            ]
        }

        self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.specific_epithet)]
        self.n_classes = sum(self.levels)
        self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.specific_epithet] for key
                        in class_list]
        self.level_names = ['family', 'subfamily', 'genus', 'specific_epithet']

        self.convert_child_of()

    def convert_child_of(self):
        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        self.child_of_family_ix, self.child_of_subfamily_ix, self.child_of_genus_ix = {}, {}, {}
        for family_name in self.child_of_family:
            if family_name not in self.family:
                continue
            self.child_of_family_ix[self.family[family_name]] = []
            for subfamily_name in self.child_of_family[family_name]:
                if subfamily_name not in self.subfamily:
                    continue
                self.child_of_family_ix[self.family[family_name]].append(self.subfamily[subfamily_name])

        for subfamily_name in self.child_of_subfamily:
            if subfamily_name not in self.subfamily:
                continue
            self.child_of_subfamily_ix[self.subfamily[subfamily_name]] = []
            for genus_name in self.child_of_subfamily[subfamily_name]:
                if genus_name not in self.genus:
                    continue
                self.child_of_subfamily_ix[self.subfamily[subfamily_name]].append(self.genus[genus_name])

        for genus_name in self.child_of_genus:
            if genus_name not in self.genus:
                continue
            self.child_of_genus_ix[self.genus[genus_name]] = []
            for genus_specific_epithet_name in self.child_of_genus[genus_name]:
                if genus_specific_epithet_name not in self.genus_specific_epithet:
                    continue
                self.child_of_genus_ix[self.genus[genus_name]].append(self.genus_specific_epithet[genus_specific_epithet_name])

        self.family_ix_to_str = {self.family[k]: k for k in self.family}
        self.subfamily_ix_to_str = {self.subfamily[k]: k for k in self.subfamily}
        self.genus_ix_to_str = {self.genus[k]: k for k in self.genus}
        self.genus_specific_epithet_ix_to_str = {self.genus_specific_epithet[k]: k for k in self.genus_specific_epithet}

    def get_one_hot(self, family, subfamily, genus, specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        retval[self.subfamily[subfamily] + self.levels[0]] = 1
        retval[self.genus[genus] + self.levels[0] + self.levels[1]] = 1
        retval[self.specific_epithet[specific_epithet] + self.levels[0] + self.levels[1] + self.levels[2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]

    def get_level_labels(self, family, subfamily, genus, specific_epithet):
        return np.array([
            self.get_label_id('family', family),
            self.get_label_id('subfamily', subfamily),
            self.get_label_id('genus', genus),
            self.get_label_id('specific_epithet', specific_epithet)
        ])

    def get_children_of(self, c_ix, level_id):
        if level_id == 0:
            # possible family
            return [self.family[k] for k in self.family]
        elif level_id == 1:
            # possible_subfamily
            return self.child_of_family_ix[c_ix]
        elif level_id == 2:
            # possible_genus
            return self.child_of_subfamily_ix[c_ix]
        elif level_id == 3:
            # possible_genus_specific_epithet
            return self.child_of_genus_ix[c_ix]
        else:
            return None

    def decode_children(self, level_labels):
        level_labels = level_labels.cpu().numpy()
        possible_family = [self.family[k] for k in self.family]
        possible_subfamily = self.child_of_family_ix[level_labels[0]]
        possible_genus = self.child_of_subfamily_ix[level_labels[1]]
        possible_genus_specific_epithet = self.child_of_genus_ix[level_labels[2]]
        new_level_labels = [
            level_labels[0],
            possible_subfamily.index(level_labels[1]),
            possible_genus.index(level_labels[2]),
            possible_genus_specific_epithet.index(level_labels[3])
        ]
        return {'family': possible_family, 'subfamily': possible_subfamily, 'genus': possible_genus,
                'genus_specific_epithet': possible_genus_specific_epithet}, new_level_labels


class ETHECLabelMapMerged(ETHECLabelMap):
    def __init__(self):
        ETHECLabelMap.__init__(self)
        self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.genus_specific_epithet)]
        self.n_classes = sum(self.levels)
        self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.genus_specific_epithet] for
                        key
                        in class_list]
        self.level_names = ['family', 'subfamily', 'genus', 'genus_specific_epithet']
        self.convert_child_of()

    def get_one_hot(self, family, subfamily, genus, genus_specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        retval[self.subfamily[subfamily] + self.levels[0]] = 1
        retval[self.genus[genus] + self.levels[0] + self.levels[1]] = 1
        retval[
            self.genus_specific_epithet[genus_specific_epithet] + self.levels[0] + self.levels[1] + self.levels[2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]

    def get_level_labels(self, family, subfamily, genus, genus_specific_epithet):
        return np.array([
            self.get_label_id('family', family),
            self.get_label_id('subfamily', subfamily),
            self.get_label_id('genus', genus),
            self.get_label_id('genus_specific_epithet', genus_specific_epithet)
        ])


class ETHEC:
    """
    ETHEC iterator.
    """

    def __init__(self, path_to_json):
        """
        Constructor.
        :param path_to_json: <str> .json path used for loading database entries.
        """
        self.path_to_json = path_to_json
        with open(path_to_json) as json_file:
            self.data_dict = json.load(json_file)
        self.data_tokens = [token for token in self.data_dict]

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> index for the entry in database
        :return: see schema.md
        """
        return self.data_dict[self.data_tokens[item]]

    def __len__(self):
        """
        Returns the number of entries in the database.
        :return: <int> Length of database
        """
        return len(self.data_tokens)

    def get_sample(self, token):
        """
        Fetch an entry based on its token.
        :param token: <str> token (uuid)
        :return: see schema.md
        """
        return self.data_dict[token]


class ETHECSmall(ETHEC):
    """
    ETHEC iterator.
    """

    def __init__(self, path_to_json, single_level=False):
        """
        Constructor.
        :param path_to_json: <str> .json path used for loading database entries.
        """
        lmap = ETHECLabelMapMergedSmall(single_level)
        self.path_to_json = path_to_json
        with open(path_to_json) as json_file:
            self.data_dict = json.load(json_file)
        # print([token for token in self.data_dict])
        if single_level:
            self.data_tokens = [token for token in self.data_dict
                                if self.data_dict[token]['family'] in lmap.family]
        else:
            self.data_tokens = [token for token in self.data_dict
                                if '{}_{}'.format(self.data_dict[token]['genus'],
                                                  self.data_dict[token]['specific_epithet'])
                                in lmap.genus_specific_epithet]


class ETHECLabelMapMergedSmall(ETHECLabelMapMerged):
    def __init__(self, single_level=False):
        self.single_level = single_level
        ETHECLabelMapMerged.__init__(self)

        self.family = {
            # "dummy1": 0,
            "Hesperiidae": 0,
            "Riodinidae": 1,
            "Lycaenidae": 2,
            "Papilionidae": 3,
            "Pieridae": 4
        }
        if self.single_level:
            print('== Using single_level data')
            self.levels = [len(self.family)]
            self.n_classes = sum(self.levels)
            self.classes = [key for class_list in [self.family] for key
                            in class_list]
            self.level_names = ['family']
        else:
            self.subfamily = {
                "Hesperiinae": 0,
                "Pyrginae": 1,
                "Nemeobiinae": 2,
                "Polyommatinae": 3,
                "Parnassiinae": 4,
                "Pierinae": 5
            }
            self.genus = {
                "Ochlodes": 0,
                "Hesperia": 1,
                "Pyrgus": 2,
                "Spialia": 3,
                "Hamearis": 4,
                "Polycaena": 5,
                "Agriades": 6,
                "Parnassius": 7,
                "Aporia": 8
            }
            self.genus_specific_epithet = {
                "Ochlodes_venata": 0,
                "Hesperia_comma": 1,
                "Pyrgus_alveus": 2,
                "Spialia_sertorius": 3,
                "Hamearis_lucina": 4,
                "Polycaena_tamerlana": 5,
                "Agriades_lehanus": 6,
                "Parnassius_jacquemonti": 7,
                "Aporia_crataegi": 8,
                "Aporia_procris": 9,
                "Aporia_potanini": 10,
                "Aporia_nabellica": 11

            }
            self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.genus_specific_epithet)]
            self.n_classes = sum(self.levels)
            self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.genus_specific_epithet]
                            for key in class_list]
            self.level_names = ['family', 'subfamily', 'genus', 'genus_specific_epithet']
            self.convert_child_of()

    def get_one_hot(self, family, subfamily, genus, genus_specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        if not self.single_level:
            retval[self.subfamily[subfamily] + self.levels[0]] = 1
            retval[self.genus[genus] + self.levels[0] + self.levels[1]] = 1
            retval[self.genus_specific_epithet[genus_specific_epithet] + self.levels[0] + self.levels[1] + self.levels[
                2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]

    def get_level_labels(self, family, subfamily=None, genus=None, genus_specific_epithet=None):
        if not self.single_level:
            return np.array([
                self.get_label_id('family', family),
                self.get_label_id('subfamily', subfamily),
                self.get_label_id('genus', genus),
                self.get_label_id('genus_specific_epithet', genus_specific_epithet)
            ])
        else:
            return np.array([
                self.get_label_id('family', family)
            ])


class ETHECDB(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset.
    """

    def __init__(self, path_to_json, path_to_images, labelmap, transform=None):
        """
        Constructor.
        :param path_to_json: <str> Path to .json from which to read database entries.
        :param path_to_images: <str> Path to parent directory where images are stored.
        :param labelmap: <ETHECLabelMap> Labelmap.
        :param transform: <torchvision.transforms> Set of transforms to be applied to the entries in the database.
        """
        self.path_to_json = path_to_json
        self.path_to_images = path_to_images
        self.labelmap = labelmap
        self.ETHEC = ETHEC(self.path_to_json)
        self.transform = transform

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'image': <np.array> image, 'labels': <np.array(n_classes)> hot vector, 'leaf_label': <int>}
        """
        sample = self.ETHEC.__getitem__(item)
        image_folder = sample['image_path'][11:21] + "R" if '.JPG' in sample['image_path'] else sample['image_name'][
                                                                                                11:21] + "R"
        path_to_image = os.path.join(self.path_to_images, image_folder,
                                     sample['image_path'] if '.JPG' in sample['image_path'] else sample['image_name'])
        img = cv2.imread(path_to_image)
        if img is None:
            print('This image is None: {} {}'.format(path_to_image, sample['token']))

        img = np.array(img)
        if self.transform:
            img = self.transform(img)

        ret_sample = {
            'image': img,
            'labels': torch.from_numpy(self.labelmap.get_one_hot(sample['family'], sample['subfamily'], sample['genus'],
                                                                 sample['specific_epithet'])).float(),
            'leaf_label': self.labelmap.get_label_id('specific_epithet', sample['specific_epithet']),
            'level_labels': torch.from_numpy(self.labelmap.get_level_labels(sample['family'], sample['subfamily'],
                                                                            sample['genus'],
                                                                            sample['specific_epithet'])).long(),
            'path_to_image': path_to_image
        }
        return ret_sample

    def __len__(self):
        """
        Return number of entries in the database.
        :return: <int> length of database
        """
        return len(self.ETHEC)

    def get_sample(self, token):
        """
        Fetch database entry based on its token.
        :param token: <str> Token used to fetch corresponding entry. (uuid)
        :return: see schema.md
        """
        return self.ETHEC.get_sample(token)


class ETHECDBMerged(ETHECDB):
    """
    Creates a PyTorch dataset.
    """

    def __init__(self, path_to_json, path_to_images, labelmap, transform=None, with_images=True):
        """
        Constructor.
        :param path_to_json: <str> Path to .json from which to read database entries.
        :param path_to_images: <str> Path to parent directory where images are stored.
        :param labelmap: <ETHECLabelMap> Labelmap.
        :param transform: <torchvision.transforms> Set of transforms to be applied to the entries in the database.
        """
        ETHECDB.__init__(self, path_to_json, path_to_images, labelmap, transform)
        self.with_images = with_images

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'image': <np.array> image, 'labels': <np.array(n_classes)> hot vector, 'leaf_label': <int>}
        """

        sample = self.ETHEC.__getitem__(item)
        if self.with_images:
            image_folder = sample['image_path'][11:21] + "R" if '.JPG' in sample['image_path'] else sample['image_name'][
                                                                                                    11:21] + "R"
            path_to_image = os.path.join(self.path_to_images, image_folder,
                                         sample['image_path'] if '.JPG' in sample['image_path'] else sample['image_name'])
            img = cv2.imread(path_to_image)
            if img is None:
                print('This image is None: {} {}'.format(path_to_image, sample['token']))

            img = np.array(img)
            if self.transform:
                img = self.transform(img)
        else:
            path_to_image, img = 0, 0

        ret_sample = {
            'image': img,
            'labels': torch.from_numpy(self.labelmap.get_one_hot(sample['family'], sample['subfamily'], sample['genus'],
                                                                 '{}_{}'.format(sample['genus'],
                                                                                sample['specific_epithet']))).float(),
            'leaf_label': self.labelmap.get_label_id('genus_specific_epithet',
                                                     '{}_{}'.format(sample['genus'], sample['specific_epithet'])),
            'level_labels': torch.from_numpy(self.labelmap.get_level_labels(sample['family'], sample['subfamily'],
                                                                            sample['genus'],
                                                                            '{}_{}'.format(sample['genus'], sample[
                                                                                'specific_epithet']))).long(),
            'path_to_image': path_to_image
        }
        return ret_sample


class ETHECDBMergedSmall(ETHECDBMerged):
    """
    Creates a PyTorch dataset.
    """

    def __init__(self, path_to_json, path_to_images, labelmap, transform=None, with_images=False):
        """
        Constructor.
        :param path_to_json: <str> Path to .json from which to read database entries.
        :param path_to_images: <str> Path to parent directory where images are stored.
        :param labelmap: <ETHECLabelMap> Labelmap.
        :param transform: <torchvision.transforms> Set of transforms to be applied to the entries in the database.
        """
        ETHECDBMerged.__init__(self, path_to_json, path_to_images, labelmap, transform, with_images )
        if hasattr(labelmap, 'single_level'):
            self.ETHEC = ETHECSmall(self.path_to_json, labelmap.single_level)
        else:
            self.ETHEC = ETHECSmall(self.path_to_json)


def generate_labelmap(path_to_json):
    """
    Generates entries for labelmap.
    :param path_to_json: <str> Path to .json to read database from.
    :return: -
    """
    ethec = ETHEC(path_to_json)
    family, subfamily, genus, specific_epithet, genus_specific_epithet = {}, {}, {}, {}, {}
    f_c, s_c, g_c, se_c, gse_c = 0, 0, 0, 0, 0
    for sample in tqdm(ethec):
        if sample['family'] not in family:
            family[sample['family']] = f_c
            f_c += 1
        if sample['subfamily'] not in subfamily:
            subfamily[sample['subfamily']] = s_c
            s_c += 1
        if sample['genus'] not in genus:
            genus[sample['genus']] = g_c
            g_c += 1
        if sample['specific_epithet'] not in specific_epithet:
            specific_epithet[sample['specific_epithet']] = se_c
            se_c += 1
        if '{}_{}'.format(sample['genus'], sample['specific_epithet']) not in genus_specific_epithet:
            genus_specific_epithet['{}_{}'.format(sample['genus'], sample['specific_epithet'])] = gse_c
            gse_c += 1
    print(json.dumps(family, indent=4))
    print(json.dumps(subfamily, indent=4))
    print(json.dumps(genus, indent=4))
    print(json.dumps(specific_epithet, indent=4))
    print(json.dumps(genus_specific_epithet, indent=4))


class SplitDataset:
    """
    Splits a given dataset to train, val and test.
    """

    def __init__(self, path_to_json, path_to_images, path_to_save_splits, labelmap, train_ratio=0.8, val_ratio=0.1,
                 test_ratio=0.1):
        """
        Constructor.
        :param path_to_json: <str> Path to .json to read database from.
        :param path_to_images: <str> Path to parent directory that contains the images.
        :param path_to_save_splits: <str> Path to directory where the .json splits are saved.
        :param labelmap: <ETHECLabelMap> Labelmap
        :param train_ratio: <float> Proportion of the dataset used for train.
        :param val_ratio: <float> Proportion of the dataset used for val.
        :param test_ratio: <float> Proportion of the dataset used for test.
        """
        if train_ratio + val_ratio + test_ratio != 1:
            print('Warning: Ratio does not add up to 1.')
        self.path_to_save_splits = path_to_save_splits
        self.path_to_json = path_to_json
        self.database = ETHEC(self.path_to_json)
        self.train_ratio, self.val_ratio, self.test_ratio = train_ratio, val_ratio, test_ratio
        self.labelmap = labelmap
        self.train, self.val, self.test = {}, {}, {}
        self.stats = {}
        self.minimum_samples = 3
        self.minimum_samples_to_use_split = 10
        print('Database has {} sample.'.format(len(self.database)))

    def collect_stats(self):
        """
        Generate counts for each class
        :return: -
        """
        for data_id in range(len(self.database)):
            sample = self.database[data_id]

            label_id = self.labelmap.get_label_id('genus_specific_epithet',
                                                  '{}_{}'.format(sample['genus'], sample['specific_epithet']))
            if label_id not in self.stats:
                self.stats[label_id] = [sample['token']]
            else:
                self.stats[label_id].append(sample['token'])
        # print({label_id: len(self.stats[label_id]) for label_id in self.stats})

    def split(self):
        """
        Split data.
        :return: -
        """
        for label_id in self.stats:
            samples_for_label_id = self.stats[label_id]
            n_samples = len(samples_for_label_id)
            if n_samples < self.minimum_samples:
                continue

            # if the number of samples are less than self.minimum_samples_to_use_split then split them equally
            if n_samples < self.minimum_samples_to_use_split:
                n_train_samples, n_val_samples, n_test_samples = n_samples // 3, n_samples // 3, n_samples // 3
            else:
                n_train_samples = int(self.train_ratio * n_samples)
                n_val_samples = int(self.val_ratio * n_samples)
                n_test_samples = int(self.test_ratio * n_samples)

            remaining_samples = n_samples - (n_train_samples + n_val_samples + n_test_samples)
            n_val_samples += remaining_samples % 2 + remaining_samples // 2
            n_test_samples += remaining_samples // 2

            # print(label_id, n_train_samples, n_val_samples, n_test_samples)

            train_samples_id_list = samples_for_label_id[:n_train_samples]
            val_samples_id_list = samples_for_label_id[n_train_samples:n_train_samples + n_val_samples]
            test_samples_id_list = samples_for_label_id[-n_test_samples:]

            for sample_id in train_samples_id_list:
                self.train[sample_id] = self.database.get_sample(sample_id)
            for sample_id in val_samples_id_list:
                self.val[sample_id] = self.database.get_sample(sample_id)
            for sample_id in test_samples_id_list:
                self.test[sample_id] = self.database.get_sample(sample_id)

    def write_to_disk(self):
        """
        Write the train, val, test .json splits to disk.
        :return: -
        """
        with open(os.path.join(self.path_to_save_splits, 'train_merged.json'), 'w') as fp:
            json.dump(self.train, fp, indent=4)
        with open(os.path.join(self.path_to_save_splits, 'val_merged.json'), 'w') as fp:
            json.dump(self.val, fp, indent=4)
        with open(os.path.join(self.path_to_save_splits, 'test_merged.json'), 'w') as fp:
            json.dump(self.test, fp, indent=4)

    def make_split_to_disk(self):
        """
        Collectively call functions to make splits and save to disk.
        :return: -
        """
        self.collect_stats()
        self.split()
        self.write_to_disk()


def generate_normalization_values(dataset):
    """
    Calculate mean and std values for a dataset.
    :param dataset: <PyTorch dataset> dataset to calculate mean, std over
    :return: -
    """

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(loader):
        batch_samples = data['image'].size(0)
        data = data['image'].view(batch_samples, data['image'].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print('Mean: {}, Std: {}'.format(mean, std))


def print_labelmap():
    path_to_json = '../database/ETHEC/'
    with open(os.path.join(path_to_json, 'train.json')) as json_file:
        data_dict = json.load(json_file)
    family, subfamily, genus, specific_epithet, genus_specific_epithet = {}, {}, {}, {}, {}
    f_c, sf_c, g_c, se_c, gse_c = 0, 0, 0, 0, 0
    # to store the children for each node
    child_of_family, child_of_subfamily, child_of_genus = {}, {}, {}
    for key in data_dict:
        if data_dict[key]['family'] not in family:
            family[data_dict[key]['family']] = f_c
            child_of_family[data_dict[key]['family']] = []
            f_c += 1
        if data_dict[key]['subfamily'] not in subfamily:
            subfamily[data_dict[key]['subfamily']] = sf_c
            child_of_subfamily[data_dict[key]['subfamily']] = []
            child_of_family[data_dict[key]['family']].append(data_dict[key]['subfamily'])
            sf_c += 1
        if data_dict[key]['genus'] not in genus:
            genus[data_dict[key]['genus']] = g_c
            child_of_genus[data_dict[key]['genus']] = []
            child_of_subfamily[data_dict[key]['subfamily']].append(data_dict[key]['genus'])
            g_c += 1
        if data_dict[key]['specific_epithet'] not in specific_epithet:
            specific_epithet[data_dict[key]['specific_epithet']] = se_c
            se_c += 1
        if '{}_{}'.format(data_dict[key]['genus'], data_dict[key]['specific_epithet']) not in genus_specific_epithet:
            genus_specific_epithet['{}_{}'.format(data_dict[key]['genus'], data_dict[key]['specific_epithet'])] = gse_c
            specific_epithet[data_dict[key]['specific_epithet']] = se_c
            child_of_genus[data_dict[key]['genus']].append(
                '{}_{}'.format(data_dict[key]['genus'], data_dict[key]['specific_epithet']))
            gse_c += 1
    print(json.dumps(family, indent=4))
    print(json.dumps(subfamily, indent=4))
    print(json.dumps(genus, indent=4))
    print(json.dumps(specific_epithet, indent=4))
    print(json.dumps(genus_specific_epithet, indent=4))

    print(json.dumps(child_of_family, indent=4))
    print(json.dumps(child_of_subfamily, indent=4))
    print(json.dumps(child_of_genus, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help='Parent directory with images.', type=str)
    parser.add_argument("--json_path", help='Path to json with relevant data.', type=str)
    parser.add_argument("--path_to_save_splits", help='Path to json with relevant data.', type=str)
    parser.add_argument("--mode", help='Path to json with relevant data. [split, calc_mean_std, small]', type=str)
    args = parser.parse_args()

    labelmap = ETHECLabelMap()
    # mean: tensor([143.2341, 162.8151, 177.2185], dtype=torch.float64)
    # std: tensor([66.7762, 59.2524, 51.5077], dtype=torch.float64)

    if args.mode == 'split':
        # create files with train, val and test splits
        data_splitter = SplitDataset(args.json_path, args.images_dir, args.path_to_save_splits, ETHECLabelMapMerged())
        data_splitter.make_split_to_disk()

    elif args.mode == 'show_labelmap':
        print_labelmap()

    elif args.mode == 'calc_mean_std':
        tform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
        train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                            path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                            labelmap=labelmap, transform=tform)
        generate_normalization_values(train_set)
    elif args.mode == 'small':
        labelmap = ETHECLabelMapMergedSmall(single_level=True)
        initial_crop = 324
        input_size = 224
        train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((initial_crop, initial_crop)),
                                                    transforms.RandomCrop((input_size, input_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    # ColorJitter(brightness=0.2, contrast=0.2),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                         std=(66.7762, 59.2524, 51.5077))])
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                            std=(66.7762, 59.2524, 51.5077))])
        train_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/train.json',
                                       path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                       labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                     path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                     labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                      labelmap=labelmap, transform=val_test_data_transforms)
        print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                                  len(test_set)))
        print(train_set[0])
    else:
        labelmap = ETHECLabelMapMerged()
        initial_crop = 324
        input_size = 224
        train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((initial_crop, initial_crop)),
                                                    transforms.RandomCrop((input_size, input_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    # ColorJitter(brightness=0.2, contrast=0.2),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                         std=(66.7762, 59.2524, 51.5077))])
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                            std=(66.7762, 59.2524, 51.5077))])
        train_set = ETHECDBMerged(path_to_json='../database/ETHEC/train.json',
                                  path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                  labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMerged(path_to_json='../database/ETHEC/val.json',
                                path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMerged(path_to_json='../database/ETHEC/test.json',
                                 path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                 labelmap=labelmap, transform=val_test_data_transforms)
        print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                                  len(test_set)))
        print(train_set[0])
