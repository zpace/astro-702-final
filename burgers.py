import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

def start(dx, xlims, state):
	x = np.arange(xlims[0], xlims[1] + dx, step = dx)
	y = np.zeros(len(x))

	if state in ['rising', 'r', 'RISING', 'R']:
		y[x > 1] = 1.
		y[np.abs(x) <= 1.] = 0.5 * (x[np.abs(x) <= 1.] + 1.)
	elif state in ['falling', 'f', 'FALLING', 'F']:
		y[x < 1] = 1.
		y[np.abs(x) <= 1.] = 0.5 * (1. - x[np.abs(x) <= 1.])
	else:
		raise ValueError('"state" requires either "rising"/"r" or "falling"/"f"')

	return x, y

def prop_FTCS(y, dt, dx):
	'''
	use FTCS scheme to propagate Burgers' eq
	'''

	#see eq 2.6 & 2.7 for description of straight extrapolation @ bdys

	y = np.concatenate((np.array([y[0]]), y, np.array([y[-1]])))
	y_new = (y[1:-1] - dt*y[1:-1] * ( ( y[2:] - y[:-2] ) / ( 2.*dx ) ) )

	return y_new

def prop_FTBS(y, dt, dx):
	'''
	use FTBS scheme to propagate Burgers' eq
	'''

	y = np.concatenate((np.array([y[0]]), y, np.array([y[-1]])))
	y_new = y[1:-1] - dt*y[1:-1] * ( ( y[1:-1] - y[:-2] ) / dx )
	
	return y_new


def prop_FTFS(y, dt, dx):
	'''
	use FTFS scheme to propagate Burgers' eq
	'''
	
	y = np.concatenate((np.array([y[0]]), y, np.array([y[-1]])))
	y_new = y[1:-1] - dt*y[1:-1] * ( ( y[2:] - y[1:-1] ) / dx )
	
	return y_new

def godunov(y, dt, dx):
	'''
	use Godunov's method to propagate Burgers' eq
	'''

	y = np.concatenate((np.array([y[0]]), y, np.array([y[-1]])))

	fp = np.zeros(len(y) - 2)
	fm = np.zeros(len(y) - 2)
	Sp = np.zeros(len(y) - 2)
	Sm = np.zeros(len(y) - 2)

	for i in range(len(fp)):
		#first build the fp array
		if np.abs(y[i+1] - y[i+2]) > 1e-10:
			#do the normal case
			Sp_i = 0.5 * (y[i+1] + y[i+2])
			if Sp_i > 0:
				fp[i] = 0.5 * y[i + 1]**2.
			elif Sp_i < 0:
				fp[i] = 0.5 * y[i + 2]**2.
			Sp[i] = Sp_i
		else: #i.e., if np.abs(y[i+1] - y[i+2]) < 1e-10:
			#do the special case  
			fp[i] = 0.5 * y[i + 1]**2.

		#now build the fm array
		if np.abs(y[i] - y[i+1]) > 1e-10:
			#do the normal case
			Sm_i = 0.5 * (y[i] + y[i+1])
			if Sm_i > 0:
				fm[i] = 0.5 * y[i]**2.
			elif Sm_i < 0:
				fm[i] = 0.5 * y[i + 1]**2.
			Sm[i] = Sm_i
		else:
			fm[i] = 0.5 * y[i]**2.

	y_new = y[1:-1] - dt/dx * (fp - fm)

	return y_new

def Q1(dx, dt, xlims, tlims):
	xr, yr = start(dx, xlims, 'r')

	xf, yf = start(dx, xlims, 'f')

	yr_0, yf_0 = yr, yf #keep the ICs in reserve

	ts = np.arange(tlims[0], tlims[1] + dt, dt)[1:]

	plt.figure(figsize = (6, 4))

	plt.plot(xr, yr_0, label = 'Rising', c = 'b', alpha = 0.5)
	plt.plot(xf, yf_0, label = 'Falling', c = 'g', alpha = 0.5)
	plt.xlabel('$x$', size = 18)
	plt.ylabel('$y$', size = 18)
	plt.legend(loc = 'best')
	plt.ylim([-0.1, 1.1])
	plt.xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
	plt.title('Initial Condition')
	plt.tight_layout()
	plt.savefig('IC.png')

	#propagate in FTBS
	for _ in ts:
		yr = prop_FTBS(yr, dt, dx)
		yf = prop_FTBS(yf, dt, dx)

	plt.figure(figsize = (6, 4))

	plt.plot(xr, yr, label = 'Rising', c = 'b', alpha = 0.5)
	plt.plot(xf, yf, label = 'Falling', c = 'g', alpha = 0.5)
	plt.xlabel('$x$', size = 18)
	plt.ylabel('$y$', size = 18)
	plt.legend(loc = 'best')
	plt.ylim([-0.1, 1.1])
	plt.xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
	plt.title('FTBS')
	plt.tight_layout()
	plt.savefig('Q1_FTBS.png') 

	yr, yf = yr_0, yf_0

	#propagate in FTCS
	for _ in ts:
		yr = prop_FTCS(yr, dt, dx)
		yf = prop_FTCS(yf, dt, dx)

	plt.figure(figsize = (6, 4))

	plt.plot(xr, yr, label = 'Rising', c = 'b', alpha = 0.5)
	plt.plot(xf, yf, label = 'Falling', c = 'g', alpha = 0.5)
	plt.xlabel('$x$', size = 18)
	plt.ylabel('$y$', size = 18)
	plt.legend(loc = 'best')
	plt.ylim([-0.1, 1.1])
	plt.xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
	plt.title('FTCS')
	plt.tight_layout()
	plt.savefig('Q1_FTCS.png')

	yr, yf = yr_0, yf_0

	#propagate in FTFS
	for _ in ts:
		yr = prop_FTFS(yr, dt, dx)
		yf = prop_FTFS(yf, dt, dx)

	plt.figure(figsize = (6, 4))

	plt.plot(xr, yr, label = 'Rising', c = 'b', alpha = 0.5)
	plt.plot(xf, yf, label = 'Falling', c = 'g', alpha = 0.5)
	plt.xlabel('$x$', size = 18)
	plt.ylabel('$y$', size = 18)
	plt.legend(loc = 'best')
	plt.ylim([-0.1, 1.1])
	plt.xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
	plt.title('FTFS')
	plt.tight_layout()
	plt.savefig('Q1_FTFS.png')

def Q2(dx, dt, xlims, tlims):
	xr, yr = start(dx, xlims, 'r')

	xf, yf = start(dx, xlims, 'f')

	yr_0, yf_0 = yr, yf #keep the ICs in reserve

	#set up history arrays to allow post-processing
	yr_h, yf_h = yr[np.newaxis, :], yf[np.newaxis, :]

	ts = np.arange(tlims[0], tlims[1] + dt, dt)

	#FTBS is the stable FD scheme, so we'll proceed with it

	plot_times = [0., 1., 2., 4.]
	plot_yn_a = np.array([ ( np.abs(ts - time) == np.min(np.abs(ts - time)) ) for time in plot_times]).sum(axis = 0) > 0

	plt.close('all')

	fig = plt.figure()
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)
	
	for time, ax in zip(plot_times, [ax1, ax2, ax3, ax4]):
		ax.set_xlabel('$x$', size = 18)
		ax.set_ylabel('$y$', size = 18)
		ax.set_ylim([-0.1, 1.1])
		ax.set_xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
		ax.set_title('$t = {}$'.format(time), size = 18)

	num_plots_done = 0

	for t, plot_yn in zip(ts, plot_yn_a):
		if t != 0.:
			yr = prop_FTBS(yr, dt, dx)
			yf = prop_FTBS(yf, dt, dx)
			yr_h = np.row_stack((yr_h, yr))
			yf_h = np.row_stack((yf_h, yf))

		axs = [ax1, ax2, ax3, ax4]

		if plot_yn == True:
			axs[num_plots_done].plot(xr, yr, label = 'Rising', c = 'b', alpha = 0.5)
			axs[num_plots_done].plot(xf, yf, label = 'Falling', c = 'g', alpha = 0.5)

			axs[num_plots_done].legend(loc = 'best')
			num_plots_done += 1

	plt.tight_layout()
	plt.savefig('Q2.png')

def Q3_4(dx, dt, xlims, tlims, test = False):
	xr, yr = start(dx, xlims, 'r')

	xf, yf = start(dx, xlims, 'f')

	yr_0, yf_0 = yr, yf #keep the ICs in reserve

	#set up history arrays to allow post-processing
	yr_h, yf_h = yr[np.newaxis, :], yf[np.newaxis, :]

	ts = np.arange(tlims[0], tlims[1] + dt, dt)

	#FTBS is the stable FD scheme, so we'll proceed with it

	plot_times = [0., 1., 2., 4.]
	plot_yn_a = np.array([ ( np.abs(ts - time) == np.min(np.abs(ts - time)) ) for time in plot_times]).sum(axis = 0) > 0

	if test == True:
		yr = godunov(yr, dt, dx)
		yf = godunov(yf, dt, dx)

		plt.close('all')

		plt.figure(figsize = (6, 4))
		plt.plot(xr, yr, label = 'Rising', c = 'b', alpha = 0.5)
		plt.plot(xf, yf, label = 'Falling', c = 'g', alpha = 0.5)
		plt.legend(loc = 'best')

		plt.xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
		plt.ylim([-0.1, 1.1])

		plt.show()
	else:

		plt.close('all')

		fig = plt.figure()
		ax1 = plt.subplot(221)
		ax2 = plt.subplot(222)
		ax3 = plt.subplot(223)
		ax4 = plt.subplot(224)
		
		for time, ax in zip(plot_times, [ax1, ax2, ax3, ax4]):
			ax.set_xlabel('$x$', size = 18)
			ax.set_ylabel('$y$', size = 18)
			ax.set_ylim([-0.1, 1.1])
			ax.set_xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
			ax.set_title('$t = {}$'.format(time), size = 18)

		num_plots_done = 0

		for t, plot_yn in zip(ts, plot_yn_a):
			if t != 0.:
				#print 'running', t
				yr, yr_prev = godunov(yr, dt, dx), yr
				#print np.sum(yr == yr_prev)
				yf, yf_prev = godunov(yf, dt, dx), yf
				#print np.sum(yf == yf_prev)
				yr_h = np.row_stack((yr_h, yr))
				yf_h = np.row_stack((yf_h, yf))

			axs = [ax1, ax2, ax3, ax4]

			if plot_yn == True:
				print 'plotting', axs[num_plots_done]
				axs[num_plots_done].plot(xr, yr, label = 'Rising', c = 'b', alpha = 0.5)
				axs[num_plots_done].plot(xf, yf, label = 'Falling', c = 'g', alpha = 0.5)

				axs[num_plots_done].legend(loc = 'best')
				num_plots_done += 1

		plt.tight_layout()
		plt.savefig('Q3.png')

	#now plot the total integrated quantity across time for each case

	Yr = np.sum(yr_h, axis = -1) * dx
	Yf = np.sum(yf_h, axis = -1) * dx

	plt.close('all')
	plt.figure(figsize = (6, 4))

	plt.plot(ts[:len(Yr)], Yr, c = 'b', alpha = 0.5, label = 'Rising')
	plt.plot(ts[:len(Yf)], Yf, c = 'g', alpha = 0.5, label = 'Falling')
	plt.legend(loc = 'best')
	plt.xlabel('$t$', size = 18)
	plt.ylabel('$Y$', size = 18)
	plt.tight_layout()
	plt.savefig('Q4.png')

def Q5(dx, xlims, tlims):
	dt = 1.25*dx

	ts = np.arange(tlims[0], tlims[1] + dt, dt)

	xr, yr = start(dx, xlims, 'r')

	xf, yf = start(dx, xlims, 'f')

	yr_0, yf_0 = yr, yf #keep the ICs in reserve

	#set up history arrays to allow post-processing
	yr_h, yf_h = yr[np.newaxis, :], yf[np.newaxis, :]

	plt.close('all')

	fig = plt.figure()
	ax = plt.subplot(111)
	ax.set_xlabel('$x$', size = 18)
	ax.set_ylabel('$y$', size = 18)
	ax.set_ylim([-0.1, 1.1])
	ax.set_xlim([xlims[0] - 0.1*xlims[1], 1.1*xlims[1]])
	ax.set_title('Instability: $dt = {}$'.format(dt) + ', $dx = {}$'.format(dx))

	num_plots_done = 0

	for t in ts:
		if t != 0.:
			#print 'running', t
			yr, yr_prev = godunov(yr, dt, dx), yr
			#print np.sum(yr == yr_prev)
			yf, yf_prev = godunov(yf, dt, dx), yf
			#print np.sum(yf == yf_prev)
			yr_h = np.row_stack((yr_h, yr))
			yf_h = np.row_stack((yf_h, yf))

		if t%1 == 0.:
			ax.plot(xr, yr, linewidth = 0.75, label = '$t = {}$'.format(t) + ', rising')
			ax.plot(xf, yf, linestyle = '--', linewidth = 0.75, label = '$t = {}$'.format(t) + ', falling')

	ax.legend(loc = 'best')

	plt.savefig('Q5.png')