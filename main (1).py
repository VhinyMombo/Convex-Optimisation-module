gamma = np.array([30,150])
ell = quadratic(gamma)
x0 = np.array([40,40])
rs1,loss1 = ell.grad_descent(x0)
rs2, loss2 = ell.newton_method(x0)

##### import numpy as np
#gamma = np.array([4,1.5])
npts = 200
xlim = 70
ylim = xlim
x1 = np.linspace(-xlim, xlim, npts)
x2 = np.linspace(-ylim, ylim, npts)
X1, X2 = np.meshgrid(x1, x2)
Y = (np.sqrt((np.array([X1.flatten(),X2.flatten()])
              .T**2*gamma)
         .sum(axis = 1))
         .reshape([npts,npts]))
plt.figure(figsize=(15,5))
ax = plt.subplot(121)
ax.plot(np.sqrt(loss1), label = 'Gradient descent')
ax.plot(np.sqrt(loss2), label = 'Steepest descent')
plt.xlabel('# iteration')
plt.ylabel('loss')
ax.legend()

#cm = plt.cm.get_cmap('viridis')
#plt.scatter(X1, X2, c=Y, cmap=cm)
#plt.xlabel('X1')
#plt.ylabel('X2')
ax2 =  plt.subplot(122)
for i in range(len(rs1) - 1):
    ax2.annotate('', xy=rs1[i + 1], xytext=rs1[i],
                 arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                 va='center', ha='center')
for i in range(len(rs2) - 1):
    ax2.annotate('', xy=rs2[i + 1], xytext=rs2[i],
                 arrowprops={'arrowstyle': '->', 'color': 'orange', 'lw': 1},
                 va='center', ha='center')
cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X1, X2, Y, )
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(["fsdf"])
#plt.legend('grad descent', 'steepest descent')
plt.show()