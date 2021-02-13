import sys

utt2spkfile = sys.argv[1]
wavscpfile = sys.argv[2]

newutt2spkfile = utt2spkfile+'.new'
newwavscpfile = wavscpfile+'.new'
uttid2spkprefixedid = {}

with open(utt2spkfile, 'r') as uttrf, open(wavscpfile, 'r') as wavrf, open(newutt2spkfile, 'w') as uttwf, open(newwavscpfile, 'w') as wavwf:
	for line in uttrf.readlines():
		uttid, spkid = line.split()
		uttid2spkprefixedid.update({uttid:spkid+'_'+uttid})
		uttwf.write('%s %s\n' %(spkid+'_'+uttid, spkid))

	for line in wavrf.readlines():
		uttid, rest = line.split(' flac')
		fullid = uttid2spkprefixedid.get(uttid)
		if fullid == None:
			raise NameError(line)
		else:
			wavwf.write('%s flac%s' %(fullid, rest))

