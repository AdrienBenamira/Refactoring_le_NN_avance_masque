import numpy as np
from os import urandom
import numpy as np
from os import urandom
import numpy as np


def WORD_SIZE():
    return (16);


def ALPHA():
    return (7);


def BETA():
    return (2);


MASK_VAL = 2 ** WORD_SIZE() - 1;






def enc_one_round(p, k):
    x, y = p[0], p[1];



    # Generate all circular shifts
    ls_1_x = ((x >> (WORD_SIZE() - 1)) + (x << 1)) & MASK_VAL
    ls_8_x = ((x >> (WORD_SIZE() - 8)) + (x << 8)) & MASK_VAL
    ls_2_x = ((x >> (WORD_SIZE() - 2)) + (x << 2)) & MASK_VAL

    # XOR Chain
    xor_1 = (ls_1_x & ls_8_x) ^ y
    xor_2 = xor_1 ^ ls_2_x
    new_x = k ^ xor_2

    return new_x, x


def dec_one_round(c, k):
    """Complete One Inverse Feistel Round
    :param x: Upper bits of current ciphertext
    :param y: Lower bits of current ciphertext
    :param k: Round Key
    :return: Upper and Lower plaintext segments
    """
    x, y = c[0], c[1];

    # Generate all circular shifts
    ls_1_y = ((y >> (WORD_SIZE() - 1)) + (y << 1)) & MASK_VAL
    ls_8_y = ((y >> (WORD_SIZE() - 8)) + (y << 8)) & MASK_VAL
    ls_2_y = ((y >> (WORD_SIZE() - 2)) + (y << 2)) & MASK_VAL

    # Inverse XOR Chain
    xor_1 = k ^ x
    xor_2 = xor_1 ^ ls_2_y
    new_x = (ls_1_y & ls_8_y) ^ xor_2
    return y, new_x


def expand_key_simon(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k) - 1];
    l = list(reversed(k[:len(k) - 1]));
    for i in range(t - 1):
        l[i % 3], ks[i + 1] = enc_one_round((l[i % 3], ks[i]), i);
    return (ks);


def encrypt_simon(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x, y = enc_one_round((x, y), k);
    return (x, y);


def decrypt_simon(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x, y), k);
    return (x, y);




def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(), len(arr[0])), dtype=np.uint8);
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return (X);


#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    
    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

#baseline training data generator
def make_train_data(n, nr, diff=(0, 0x0040)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key_simon(keys, nr);
  ctdata0l, ctdata0r = encrypt_simon((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simon((plain1l, plain1r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

#real differences data generator
def real_differences_data(n, nr, diff=(0, 0x0040)):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  #generate keys
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  #generate plaintexts
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #apply input difference
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  #expand keys and encrypt
  ks = expand_key_simon(keys, nr);
  ctdata0l, ctdata0r = encrypt_simon((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simon((plain1l, plain1r), ks);
  #generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);
