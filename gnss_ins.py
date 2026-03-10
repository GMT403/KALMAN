import numpy as np
import plotly.graph_objects as go
from google.colab import files
from datetime import datetime, timedelta
from pymavlink import mavutil

# ==============================
# 1️⃣ EKF + INS Sınıfı (INS mekanizasyonu güncel)
# ==============================
class EKF_INS:
    def __init__(self, init_pos, init_vel, init_rpy):
        self.x = np.zeros((15,1))
        self.x[0:3] = init_pos.reshape(3,1)
        self.x[3:6] = init_vel.reshape(3,1)
        self.P = np.eye(15)*0.1
        self.P[6:9,6:9]=np.eye(3)*np.deg2rad(5)
        self.P[9:15,9:15]=np.eye(6)*0.5
        self.Q=np.eye(15)*0.01
        self.R=np.diag([0.5,0.5,2.0])
        self.Rb2l = self.euler_to_rot(init_rpy[0], init_rpy[1], init_rpy[2])
        self.omega_e = 7.292115e-5
        self.a=6378137.0
        self.f=1/298.257223563
        self.e_sq=self.f*(2-self.f)

    # --- Temel Fonksiyonlar ---
    def euler_to_rot(self,r,p,y):
        cr,sr=np.cos(r),np.sin(r)
        cp,sp=np.cos(p),np.sin(p)
        cy,sy=np.cos(y),np.sin(y)
        return np.array([
            [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
            [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
            [-sp, sr*cp, cr*cp]
        ])
    
    def dcm_to_euler(self,R):
        pitch=np.arctan2(-R[2,0], np.sqrt(R[0,0]**2+R[1,0]**2))
        roll=np.arctan2(R[2,1],R[2,2])
        yaw=np.arctan2(R[1,0],R[0,0])
        return roll,pitch,yaw

    def skew(self,v):
        v=v.flatten()
        return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    
    def gravity(self, lat, h):
        g0=9.780327*(1+0.0053024*np.sin(lat)**2-0.0000058*np.sin(2*lat)**2)
        return np.array([0,0,-(g0-3.086e-6*h)])
    
    def earth_radii(self,lat):
        sin_lat=np.sin(lat)
        denom=np.sqrt(1-self.e_sq*sin_lat**2)
        Rn=self.a/denom
        Rm=self.a*(1-self.e_sq)/(denom**3)
        return Rm,Rn

    # --- INS Mekanizasyonu (geliştirilmiş) ---
    def ins_predict(self, acc, gyr, dt):
        pos = self.x[0:3].flatten()
        vel = self.x[3:6].flatten()
        lat, lon, h = pos

        # Dünya ve Coriolis etkileri
        Rm,Rn=self.earth_radii(lat)
        omega_ie_l = np.array([0,self.omega_e*np.cos(lat),self.omega_e*np.sin(lat)])
        omega_el_l = np.array([-vel[1]/(Rm+h), vel[0]/(Rn+h), vel[0]*np.tan(lat)/(Rn+h)])
        omega_il_b = self.Rb2l.T@(omega_ie_l+omega_el_l)
        omega_lb_b = gyr-omega_il_b
        
        # Dönüşüm Matrisi Güncellemesi (DCM)
        self.Rb2l = self.Rb2l@(np.eye(3)+self.skew(omega_lb_b)*dt)
        # Orto-normalizasyon
        u,s,vh = np.linalg.svd(self.Rb2l)
        self.Rb2l = u@vh
        
        # Hız ve konum
        f_l = self.Rb2l@acc
        coriolis = np.cross((2*omega_ie_l+omega_el_l),vel)
        vel += (f_l - coriolis + self.gravity(lat,h))*dt

        lat_dot = vel[1]/(Rm+h)
        lon_dot = vel[0]/((Rn+h)*np.cos(lat))
        h_dot = vel[2]
        pos[0] += lat_dot*dt
        pos[1] += lon_dot*dt
        pos[2] += h_dot*dt

        self.x[0:3] = pos.reshape(3,1)
        self.x[3:6] = vel.reshape(3,1)
        r,p,y = self.dcm_to_euler(self.Rb2l)
        self.x[6:9] = np.array([r,p,y]).reshape(3,1)

    # --- EKF Predict ---
    def ekf_predict(self, acc, gyr, dt):
        v=self.x[3:6]; att=self.x[6:9].flatten()
        acc_b=self.x[9:12]; gyr_b=self.x[12:15]
        R=self.euler_to_rot(att[0],att[1],att[2])
        acc_corr = acc.reshape(3,1)-acc_b
        gyr_corr = gyr.reshape(3,1)-gyr_b
        acc_n=R@acc_corr
        acc_n[2,0]-=9.80665
        self.x[0:3]+=v*dt+0.5*acc_n*dt**2
        self.x[3:6]+=acc_n*dt
        self.x[6:9]+=gyr_corr*dt
        F=np.eye(15)
        F[0:3,3:6]=np.eye(3)*dt
        F[3:6,6:9]=-R@self.skew(acc_corr)*dt
        F[3:6,9:12]=-R*dt
        F[6:9,12:15]=-R*dt
        self.P=F@self.P@F.T+self.Q

    # --- GPS Update ---
    def update_gps(self,z):
        H=np.zeros((3,15)); H[:,0:3]=np.eye(3)
        y=z.reshape(3,1)-H@self.x
        S=H@self.P@H.T+self.R
        K=self.P@H.T@np.linalg.inv(S)
        self.x+=K@y
        self.P=(np.eye(15)-K@H)@self.P

# ==============================
# 2️⃣ Veri yükleme ve INS+EKF çalıştırma
# ==============================
uploaded = files.upload()
filename = next(iter(uploaded))
mlog = mavutil.mavlink_connection(filename)

imu_data,gps_data=[],[]
while True:
    m=mlog.recv_match(type=['IMU','GPS'])
    if m is None: break
    d=m.to_dict()
    if m.get_type()=='IMU':
        imu_data.append([d['TimeUS'],d['AccX'],d['AccY'],d['AccZ'],d['GyrX'],d['GyrY'],d['GyrZ']])
    elif m.get_type()=='GPS' and d.get('Status',0)>=3:
        gps_data.append([d['TimeUS'],d['Lat'],d['Lng'],d['Alt']])
imu_data=np.array(imu_data); gps_data=np.array(gps_data)

# ==============================
# 3️⃣ EKF+INS Çalıştırma ve Görselleştirme
# ==============================
if len(gps_data)>0:
    lat0, lon0, alt0 = gps_data[0,1], gps_data[0,2], gps_data[0,3]
    ekf_ins = EKF_INS(np.array([0,0,0]), np.array([0,0,0]), np.array([0,0,0]))
    res=[]
    gps_plot=[]
    gps_idx=0
    last_t=imu_data[0,0]

    for i in range(len(gps_data)):
        dx = (gps_data[i,2]-lon0)*(40075000*np.cos(np.deg2rad(lat0))/360)
        dy = (gps_data[i,1]-lat0)*(40008000/360)
        dz = gps_data[i,3]-alt0
        gps_plot.append([dx,dy,dz])
    gps_plot=np.array(gps_plot)

    for imu in imu_data:
        dt=(imu[0]-last_t)/1e6
        if dt<=0 or dt>0.1: last_t=imu[0]; continue
        ekf_ins.ins_predict(imu[1:4],imu[4:7],dt)
        ekf_ins.ekf_predict(imu[1:4],imu[4:7],dt)
        if gps_idx<len(gps_data) and gps_data[gps_idx,0]<=imu[0]:
            z_gps=gps_plot[gps_idx]
            ekf_ins.update_gps(z_gps)
            gps_idx+=1
        res.append(ekf_ins.x.flatten())
        last_t=imu[0]

    res=np.array(res)
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=res[:,0],y=res[:,1],z=res[:,2],
                               mode='lines',name='INS+EKF',
                               line=dict(color='cyan',width=4)))
    fig.add_trace(go.Scatter3d(x=gps_plot[:,0],y=gps_plot[:,1],z=gps_plot[:,2],
                               mode='markers+lines',name='Ham GPS',
                               marker=dict(size=2,color='red'),
                               line=dict(color='red',width=2,dash='dash')))
    fig.update_layout(title="INS + 15-State EKF vs Raw GPS",
                      scene=dict(xaxis_title='Doğu [m]',
                                 yaxis_title='Kuzey [m]',
                                 zaxis_title='Altimetre [m]',
                                 aspectmode='data'),
                      template="plotly_dark")
    fig.show()
