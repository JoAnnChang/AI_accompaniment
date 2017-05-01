import os, random
from midi_to_statematrix import *
from data import *
import cPickle as pickle

import signal

batch_width = 10 # number of sequences in a batch
batch_len = 16*8 # length of each sequence
division_len = 16 # interval between possible start locations



def loadPieces(dirpath_main, dirpath_accomp):
    ### EDIT ###
    ### pieces[] to pieces[2][] ###

    pieces = {}

    ### main melody and accompaniment###
    for fname in os.listdir(dirpath_main):
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]
        outMatrix_m = midiToNoteStateMatrix(os.path.join(dirpath_main, fname))
        outMatrix_a = midiToNoteStateMatrix(os.path.join(dirpath_accomp, fname))
        if len(outMatrix_m) < batch_len or len(outMatrix_a) < batch_len:
            continue

        pieces[name] = [outMatrix_m, outMatrix_a]
        print "Loaded main {}".format(name)
    return pieces

def getPieceSegment(pieces):
    ### EDIT ###
    piece_choose = random.choice(pieces.keys())
    print piece_choose
    piece_output = pieces.get(piece_choose)

    #piece_output = random.choice(pieces.values())
    
    start = random.randrange(0,len(piece_output[0])-batch_len,division_len)
    #print "Range is {} {} {} -> {}".format(0,len(piece_output)-batch_len,division_len, start)
    #seg_out = piece_output[start:start+batch_len]
    #seg_in = noteStateMatrixToInputForm(seg_out)
    seg_main = piece_output[0][start:start+batch_len]
    seg_accomp = piece_output[1][start:start+batch_len]
    seg_main = noteStateMatrixToInputForm(seg_main)
    return seg_main, seg_accomp
    #return seg_in, seg_out

def getPieceBatch(pieces):
    ### EDIT ###
    i,o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return numpy.array(i), numpy.array(o)

def trainPiece(model,pieces,epochs,start=0):
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start,start+epochs):
        if stopflag[0]:
            break
        error = model.update_fun(*getPieceBatch(pieces))
        print "epoch ", i, error
	if i % 100 == 0:
            print "epoch {}, error={}".format(i,error)
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(batch_len, 1, xIpt[0])), axis=0),'output/sample{}'.format(i))
            pickle.dump(model.learned_config,open('output/params{}.p'.format(i), 'wb'))
    signal.signal(signal.SIGINT, old_handler)
