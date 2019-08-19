	FUNCTION TRAPEZ(Y,N,DX)
c	FINDS AREA BELOW CURVE (X,Y) ACCORDING TO TRAPEZOIDAL RULE.
c	ASSUMES EQUAL SPACING DX BETWEEN N POINTS.
	REAL Y(N)
	TRAPEZ=0.
	IF (N.LE.1) RETURN
	TRAPEZ=(Y(1) + Y(N))*DX/2.
	IF (N.EQ.2) RETURN
	A=0.	
	DO I=2,N-1
	A = A + Y(I)
	ENDDO
	TRAPEZ = TRAPEZ + A*DX
	RETURN
	END
