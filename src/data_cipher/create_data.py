import numpy as np
import sys

class Create_data_binary:


    def __init__(self, args, cipher, rng):
        self.args = args
        self.cipher = cipher
        self.rng = rng
        self.WORD_SIZE = self.args.word_size
        if args.cipher == "speck":
            self.diff = (0x0040, 0)
        if args.cipher == "simon":
            self.diff = (0, 0x0040)
        if args.cipher == "simeck":
            self.diff = (0x0, 0x2)
        if args.cipher == "aes228":
            self.diff = (1, 0, 0, 1)
        if args.cipher == "aes224":
            self.diff = (1, 0, 0, 1)


    def urandom_from_random(self, length):
        if length == 0:
            return b''
        chunk_size = 65535
        chunks = []
        while length >= chunk_size:
            chunks.append(self.rng.getrandbits(
                    chunk_size * 8).to_bytes(chunk_size, sys.byteorder))
            length -= chunk_size
        if length:
            chunks.append(self.rng.getrandbits(
                    length * 8).to_bytes(length, sys.byteorder))
        result = b''.join(chunks)
        return result


    def convert_to_binary(self, arr):
        X = np.zeros((len(arr) * self.WORD_SIZE, len(arr[0])), dtype=np.uint8);
        for i in range(len(arr) * self.WORD_SIZE):
            index = i // self.WORD_SIZE;
            offset = self.WORD_SIZE - (i % self.WORD_SIZE) - 1;
            X[i] = (arr[index] >> offset) & 1;
        X = X.transpose();
        return (X);


    def make_data(self, n):
        if self.args.cipher != "aes224" and self.args.cipher != "aes228" :
            X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r = self.make_train_data_general(n)
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);


    def make_train_data_general(self, n):
        Y = np.frombuffer(self.urandom_from_random(n), dtype=np.uint8);
        Y = Y & 1;
        keys = np.frombuffer(self.urandom_from_random(8 * n), dtype=np.uint16).reshape(4, -1);
        plain0l = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain0r = np.frombuffer(self.urandom_from_random(2 * n), dtype=np.uint16);
        plain1l = plain0l ^ self.diff[0];
        plain1r = plain0r ^ self.diff[1];
        num_rand_samples = np.sum(Y == 0);
        if self.args.type_create_data == "normal":
            plain1l[Y == 0] = np.frombuffer(self.urandom_from_random( 2 * num_rand_samples), dtype=np.uint16);
            plain1r[Y == 0] = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
        ks = self.cipher.expand_key(keys, self.args.nombre_round_eval);
        ctdata0l, ctdata0r = self.cipher.encrypt((plain0l, plain0r), ks);
        ctdata1l, ctdata1r = self.cipher.encrypt((plain1l, plain1r), ks);
        if self.args.type_create_data == "real_difference":
            k0 = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
            k1 = np.frombuffer(self.urandom_from_random(2 * num_rand_samples), dtype=np.uint16);
            ctdata0l[Y == 0] = ctdata0l[Y == 0] ^ k0;
            ctdata0r[Y == 0] = ctdata0r[Y == 0] ^ k1;
            ctdata1l[Y == 0] = ctdata1l[Y == 0] ^ k0;
            ctdata1r[Y == 0] = ctdata1r[Y == 0] ^ k1;
        liste_inputs = self.convert_data_inputs(self.args, ctdata0l, ctdata0r, ctdata1l, ctdata1r)
        X = self.convert_to_binary(liste_inputs);
        return (X, Y, ctdata0l, ctdata0r, ctdata1l, ctdata1r);


    def convert_data_inputs(self, args, ctdata0l, ctdata0r, ctdata1l, ctdata1r):
        inputs_toput = []
        V0 = self.cipher.ror(ctdata0l ^ ctdata0r, self.cipher.BETA)
        V1 = self.cipher.ror(ctdata1l ^ ctdata1r, self.cipher.BETA)
        DV = V0 ^ V1
        V0Inv = 65535 - V0
        V1Inv = 65535 - V1
        inv_DeltaV = 65535 - DV
        for i in range(len(args.inputs_type)):
            if args.inputs_type[i] =="ctdata0l":
                inputs_toput.append(ctdata0l)
            if args.inputs_type[i] =="ctdata1l":
                inputs_toput.append(ctdata1l)
            if args.inputs_type[i] =="ctdata0r":
                inputs_toput.append(ctdata0r)
            if args.inputs_type[i] =="ctdata1r":
                inputs_toput.append(ctdata1r)
            if args.inputs_type[i] =="V0&V1":
                inputs_toput.append(V0&V1)
            if args.inputs_type[i] =="V0|V1":
                inputs_toput.append(V0 | V1)
            if args.inputs_type[i] =="ctdata0l^ctdata1l":
                inputs_toput.append(ctdata0l^ctdata1l)
            if args.inputs_type[i] =="ctdata0l^ctdata0r":
                inputs_toput.append(ctdata0l^ctdata0r)
            if args.inputs_type[i] =="ctdata0r^ctdata1r":
                inputs_toput.append(ctdata0r^ctdata1r)
            if args.inputs_type[i] =="ctdata1l^ctdata1r":
                inputs_toput.append(ctdata1l^ctdata1r)
            if args.inputs_type[i] =="ctdata1l^ctdata0r":
                inputs_toput.append(ctdata1l^ctdata0r)
            if args.inputs_type[i] =="ctdata1r^ctdata0l":
                inputs_toput.append(ctdata1r^ctdata0l)
            if args.inputs_type[i] =="ctdata0l^ctdata1l^ctdata0r^ctdata1r":
                inputs_toput.append(ctdata0l^ctdata1l^ctdata0r^ctdata1r)
            if args.inputs_type[i] =="ctdata0r^ctdata1r^ctdata0l^ctdata1l":
                inputs_toput.append(ctdata0r^ctdata1r^ctdata0l^ctdata1l)
            if args.inputs_type[i] =="inv(V0)&V1":
                inputs_toput.append(V1&V0Inv)
            if args.inputs_type[i] =="V0&inv(V1)":
                inputs_toput.append(V0&V1Inv)
            if args.inputs_type[i] =="inv(V0)&inv(V1)":
                inputs_toput.append(V0Inv&V1Inv)
            if args.inputs_type[i] =="inv(DeltaL)":
                inv_DeltaL = 65535 - ctdata0l ^ ctdata1l
                inputs_toput.append(inv_DeltaL)
            if args.inputs_type[i] =="inv(DeltaV)":
                inv_DeltaV = 65535 - ctdata0l^ctdata1l^ctdata0r^ctdata1r
                inputs_toput.append(inv_DeltaV)
            if args.inputs_type[i] == "DeltaL&DeltaV":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL&DV)
            if args.inputs_type[i] == "DLi":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL)
            if args.inputs_type[i] == "DLi-1":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL>>1)
            if args.inputs_type[i] == "DLi+1":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL<<1)
            if args.inputs_type[i] == "DVi":
                inputs_toput.append(DV)
            if args.inputs_type[i] == "DVi-1":
                inputs_toput.append(DV>>1)
            if args.inputs_type[i] == "DVi+1":
                inputs_toput.append(DV<<1)
            if args.inputs_type[i] == "V0i":
                inputs_toput.append(V0)
            if args.inputs_type[i] == "V0i-1":
                #V0 = ctdata0l ^ ctdata0r
                inputs_toput.append(V0>>1)
            if args.inputs_type[i] == "V0i+1":
                #V0 = ctdata0l ^ ctdata0r
                inputs_toput.append(V0<<1)
            if args.inputs_type[i] == "V1i":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1)
            if args.inputs_type[i] == "V1i-1":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1>>1)
            if args.inputs_type[i] == "V1i+1":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1<<1)
            if args.inputs_type[i] == "DL":
                DeltaL = ctdata0l ^ ctdata1l
                inputs_toput.append(DeltaL)
            if args.inputs_type[i] == "inv(DL)":
                DeltaL = 65535 - (ctdata0l ^ ctdata1l)
                inputs_toput.append(DeltaL)
            if args.inputs_type[i] == "V0":
                #V0 = ctdata0l ^ ctdata0r
                inputs_toput.append(V0)
            if args.inputs_type[i] == "inv(V0)":
                #V0 = 65535 - (ctdata0l ^ ctdata0r)
                inputs_toput.append(V0Inv)
            if args.inputs_type[i] == "V1":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(V1)
            if args.inputs_type[i] == "inv(V1)":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(V1Inv)
            if args.inputs_type[i] == "DV":
                #V1 = ctdata1l ^ ctdata1r
                inputs_toput.append(DV)
            if args.inputs_type[i] == "inv(DV)":
                #V1 = 65535 - (ctdata1l ^ ctdata1r)
                inputs_toput.append(inv_DeltaV)
        return inputs_toput



