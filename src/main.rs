
use std::io::Write;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread::JoinHandle;
use std::collections::HashMap;

use ggez::{Context, GameResult, ContextBuilder};
use ggez::graphics::{self, Color as GgezColor, Rect};
use ggez::event::{self, EventHandler, KeyMods};

use rand::{thread_rng, Rng};
use rand::rngs::ThreadRng;

const EPSILON: f32 = 0.000001;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

#[derive(Debug, Clone, Copy)]
struct Tuple {
    d: [f32; 4],
}

impl Tuple {
    fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Tuple {d: [x, y, z, w]}
    }

    fn is_point(&self) -> bool {
        approx(self.w(), 1.0)
    }

    fn is_vector(&self) -> bool {
        approx(self.w(), 0.0)
    }

    fn x(&self) -> f32 {
        self.d[0]
    }

    fn y(&self) -> f32 {
        self.d[1]
    }

    fn z(&self) -> f32 {
        self.d[2]
    }

    fn w(&self) -> f32 {
        self.d[3]
    }

    fn set_w(&mut self, v: f32) {
        self.d[3] = v;
    }

    fn red(&self) -> f32 {
        self.d[0]
    }

    fn green(&self) -> f32 {
        self.d[1]
    }

    fn blue(&self) -> f32 {
        self.d[2]
    }

    fn at(&self, i: usize) -> f32 {
        self.d[i]
    }

    fn set_at(&mut self, i: usize, val: f32) {
        self.d[i] = val;
    }

    fn reflect(&self, n: Vector) -> Vector {
        *self - n * 2.0 * dot(*self, n)
    }
}

type Vector = Tuple;
type Point = Tuple;
type Color = Tuple;

fn tuple(x: f32, y: f32, z: f32, w:f32) -> Tuple {
    Tuple::new(x, y, z, w)
}

fn point(x: f32, y: f32, z: f32) -> Point {
    tuple(x, y, z, 1.0)
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    tuple(x, y, z, 0.0)
}

fn color(r: f32, g: f32, b: f32) -> Color {
    point(r, g, b)
}

fn zero() -> Vector {
    vector(0.0, 0.0, 0.0)
}

impl Add<Tuple> for Tuple {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            d: [self.d[0] + other.d[0],
                self.d[1] + other.d[1],
                self.d[2] + other.d[2],
                self.d[3] + other.d[3],
            ]
        }
    }
}

impl Sub<Tuple> for Tuple {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            d: [self.d[0] - other.d[0],
                self.d[1] - other.d[1],
                self.d[2] - other.d[2],
                self.d[3] - other.d[3],
            ]
        }
    }
}

impl Mul<f32> for Tuple {
    type Output = Self;

    fn mul(self, other: f32) -> Self {
        Self {
            d: [self.d[0] * other,
                self.d[1] * other,
                self.d[2] * other,
                self.d[3] * other,
            ]
        }
    }
}

impl Div<f32> for Tuple {
    type Output = Self;

    fn div(self, other: f32) -> Self {
        Self {
            d: [self.d[0] / other,
                self.d[1] / other,
                self.d[2] / other,
                self.d[3] / other,
            ]
        }
    }
}

impl Neg for Tuple {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            d: [-self.d[0],
                -self.d[1],
                -self.d[2],
                -self.d[3],
            ]
        }
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        approx(self.d[0], other.d[0]) &&
            approx(self.d[1], other.d[1]) &&
            approx(self.d[2], other.d[2]) &&
            approx(self.d[3], other.d[3])
    }
}

fn add(a: Tuple, b: Tuple) -> Tuple {
    a + b
//    tuple(a.d[0] + b.d[0], a.d[1] + b.d[1], a.d[2] + b.d[2], a.d[3] + b.d[3])
}

fn sub(a: Tuple, b: Tuple) -> Tuple {
    a - b
//    tuple(a.d[0] - b.d[0], a.d[1] - b.d[1], a.d[2] - b.d[2], a.d[3] - b.d[3])
}

fn negate(a: Tuple) -> Tuple {
    -a
//    tuple(-a.d[0], -a.d[1], -a.d[2], -a.d[3])
}

fn mul(a: Tuple, d: f32) -> Tuple {
    a * d
//    tuple(a.d[0] * d, a.d[1] * d, a.d[2] * d, a.d[3] * d)
}

fn div(a: Tuple, d: f32) -> Tuple {
    a / d
//    tuple(a.d[0] / d, a.d[1] / d, a.d[2] / d, a.d[3] / d)
}

fn magnitude(a: Tuple) -> f32 {
    (a.d[0] * a.d[0] + a.d[1] * a.d[1] + a.d[2] * a.d[2] + a.d[3] * a.d[3]).sqrt()
}

fn normalize(a: Tuple) -> Tuple {
    let m = magnitude(a);
    tuple(a.d[0] / m, a.d[1] / m, a.d[2] / m, a.d[3] / m)
}

fn dot(a: Tuple, b: Tuple) -> f32 {
    a.d[0] * b.d[0] + a.d[1] * b.d[1] + a.d[2] * b.d[2] + a.d[3] * b.d[3]
}

fn cross(a: Tuple, b: Tuple) -> Tuple {
    vector(a.y() * b.z() - a.z() * b.y(),
    a.z() * b.x() - a.x() * b.z(),
    a.x() * b.y() - a.y() * b.x())
}

fn hadamard_product(a: Color, b: Color) -> Tuple {
    tuple(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(), a.w() * b.w())
}

struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<Color>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Canvas {
        let pixels = vec![color(0.0, 0.0, 0.0); width * height];
        Canvas {
            width,
            height,
            pixels,
        }
    }

    fn pixel_at(&self, x: usize, y: usize) -> Color {
        self.pixels[y * self.width + x]
    }

    fn write_pixel(&mut self, x: usize, y: usize, c: Color) {
        if x >= self.width || y >= self.height {
            return;
        }
        self.pixels[y * self.width + x] = c;
    }

    fn to_ppm(&self) -> PPM {
        let mut out: Vec<u8> = Vec::new();

        write!(out, "P3\n{} {}\n255\n", self.width, self.height).expect("write to vector");

        for (i, c) in self.pixels.iter().enumerate() {
            let r = component_to_ppm(c.red());
            let g = component_to_ppm(c.green());
            let b = component_to_ppm(c.blue());
            let sep = if i == 0 {
                ""
            } else if i % 5 == 0 {
                "\n"
            } else
            { " " };
            write!(out, "{}{} {} {}", sep, r, g, b).expect("write to vector");

        }
        write!(out, "\n").expect("write to vector");
        PPM {content: out}
    }
}

fn component_to_ppm(c: f32) -> u8{
    (c.max(0.0).min(1.0) * 255.0) as u8
}

struct PPM {
    content: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct Matrix {
    size: usize,
    data: [f32; 16],
}

impl Matrix {
    #[allow(dead_code)]
    fn new(size: usize) -> Matrix {
        let data = [0.0; 16];
        Matrix {
            size,
            data,
        }
    }

    fn from_slice(slice: &[f32]) -> Matrix {
        let mut size = 1;
        while size * size < slice.len() {
            size += 1;
        }
        assert_eq!(size * size, slice.len());
        let mut data = [0.0; 16];
        slice.iter()
            .enumerate()
            .for_each(|(i, x)| { data[i] = *x; } );
        Matrix {
            size,
            data,
        }
    }
    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.size + c]
    }

    fn set_at(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.size + c] = v;
    }

    fn identity() -> Matrix {
        Matrix::from_slice(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    fn transpose(&self) -> Matrix {
        let mut res = Matrix::new(self.size);
        for r in 0..self.size {
            for c in 0..self.size {
                res.set_at(r, c, self.at(c, r));
            }
        }
        res
    }

    fn determinant(&self) -> f32 {
        if self.size == 2 {
            self.at(0, 0) * self.at(1, 1) - self.at(0, 1) * self.at(1, 0)
        } else {
            let mut val = 0.0;
            for c in 0..self.size {
                val += self.at(0, c) * self.cofactor(0, c);
            }
            val
        }
    }

    fn submatrix(&self, row: usize, col: usize) -> Matrix {
        let mut m = Matrix::new(self.size - 1);
        for r in 0..m.size {
            for c in 0..m.size {
                m.set_at(r, c,
                         self.at(if r < row { r } else { r + 1 },
                                if c < col { c } else { c + 1 }));
            }
        }
        m
    }

    fn minor(&self, row: usize, col: usize) -> f32 {
        assert_eq!(self.size, 3);
        let b = self.submatrix(row, col);
        b.determinant()
    }

    fn cofactor(&self, row: usize, col: usize) -> f32 {
        let d = self.submatrix(row, col).determinant();
        if (row + col) % 2 == 0 {
            d
        } else {
            -d
        }
    }

    fn invertible(&self) -> bool {
        self.determinant() != 0.0
    }

    fn inverse(&self) -> Matrix {
        assert!(self.invertible());
        let det = self.determinant();
        let mut m = Matrix::new(self.size);
        for row in 0..self.size {
            for col in 0..self.size {
               let c = self.cofactor(row, col);
                m.set_at(col, row, c / det);
            }
        }
        m
    }

    #[allow(dead_code)]
    fn dump(&self) {
        if self.size != 4 {
            return;
        }
        println!("/ {:12}{:12}{:12}{:12} \\", self.data[0], self.data[1], self.data[2], self.data[3]);
        println!("| {:12}{:12}{:12}{:12} |", self.data[4], self.data[5], self.data[6], self.data[7]);
        println!("| {:12}{:12}{:12}{:12} |", self.data[8], self.data[9], self.data[10], self.data[11]);
        println!("\\ {:12}{:12}{:12}{:12} /", self.data[12], self.data[13], self.data[14], self.data[15]);
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        assert_eq!(self.size, other.size);

        let mut res = Matrix::new(self.size);
        for r in 0..self.size {
            for c in 0..self.size {
                let mut val = 0.0;
                for i in 0..self.size {
                    val += self.at(r, i) * other.at(i, c);
                }
                res.set_at(r, c, val);
            }
        }
        res
    }
}

impl Mul<Tuple> for Matrix {
    type Output = Tuple;

    fn mul(self, other: Tuple) -> Tuple {
        assert_eq!(self.size, 4);

        let mut res = tuple(0.0, 0.0, 0.0, 0.0);
        for r in 0..self.size {
            let mut val = 0.0;
            for c in 0..self.size {
                val += self.at(r, c) * other.at(c);
            }
            res.set_at(r, val);
        }
        res
    }

}
impl PartialEq for Matrix {

    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        for i in 0..self.data.len() {
            if !approx(self.data[i], other.data[i]) {
                return false;
            }
        }
        true
    }
}
fn translation(x: f32, y: f32, z: f32) -> Matrix {
    let mut m = Matrix::new(4);
    m.set_at(0, 0, 1.0);
    m.set_at(1, 1, 1.0);
    m.set_at(2, 2, 1.0);
    m.set_at(3, 3, 1.0);
    m.set_at(0, 3, x);
    m.set_at(1, 3, y);
    m.set_at(2, 3, z);
    m
}

fn scaling(x: f32, y: f32, z: f32) -> Matrix {
    let mut m = Matrix::new(4);
    m.set_at(0, 0, x);
    m.set_at(1, 1, y);
    m.set_at(2, 2, z);
    m.set_at(3, 3, 1.0);
    m
}

fn rotation_x(rad: f32) -> Matrix {
    let mut m = Matrix::new(4);
    m.set_at(0, 0, 1.0);
    m.set_at(1, 1, rad.cos());
    m.set_at(2, 1, rad.sin());
    m.set_at(1, 2, -rad.sin());
    m.set_at(2, 2, rad.cos());
    m.set_at(3, 3, 1.0);
    m
}

fn rotation_y(rad: f32) -> Matrix {
    let mut m = Matrix::new(4);
    m.set_at(0, 0, rad.cos());
    m.set_at(0, 2, rad.sin());
    m.set_at(1, 1, 1.0);
    m.set_at(2, 0, -rad.sin());
    m.set_at(2, 2, rad.cos());
    m.set_at(3, 3, 1.0);
    m
}

fn rotation_z(rad: f32) -> Matrix {
    let mut m = Matrix::new(4);
    m.set_at(0, 0, rad.cos());
    m.set_at(0, 1, -rad.sin());
    m.set_at(1, 0, rad.sin());
    m.set_at(1, 1, rad.cos());
    m.set_at(2, 2, 1.0);
    m.set_at(3, 3, 1.0);
    m
}

fn shearing(x_y: f32, x_z: f32, y_x: f32, y_z: f32, z_x: f32, z_y: f32) -> Matrix {
    let mut m = Matrix::new(4);
    m.set_at(0, 0, 1.0);
    m.set_at(0, 1, x_y);
    m.set_at(0, 2, x_z);
    m.set_at(1, 0, y_x);
    m.set_at(1, 1, 1.0);
    m.set_at(1, 2, y_z);
    m.set_at(2, 0, z_x);
    m.set_at(2, 1, z_y);
    m.set_at(2, 2, 1.0);
    m.set_at(3, 3, 1.0);
    m
}

#[derive(Debug)]
struct Ray {
    origin: Point,
    direction: Vector,
}

impl Ray {
    fn new(origin: Point, direction: Vector) -> Ray {
        Ray { origin, direction, }
    }

    fn position(&self, t: f32) -> Point {
        self.origin + self.direction * t
    }

    fn transform(&self, m: &Matrix) -> Self {
        Ray { origin: *m * self.origin, direction: *m * self.direction, }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Prim {
    Sphere,
    #[allow(dead_code)]
    Unused,
}

#[derive(Debug, PartialEq, Clone)]
struct Shape {
    id: usize,
    prim: Prim,
    trans: Matrix,
    mat: Material,
}

fn sphere(id: usize) -> Shape {
    Shape {
        id: id,
        prim: Prim::Sphere,
        trans: Matrix::identity(),
        mat: Material::default(),
    }
}

impl Shape {
    fn set_transform(&mut self, m: Matrix) {
        assert_eq!(m.size, 4);
        self.trans = m;
    }

    fn transform(&self) -> Matrix {
        self.trans
    }

    fn set_material(&mut self, m: Material) {
        self.mat = m;

    }

    fn material(&self) -> &Material {
        &self.mat
    }

    fn normal_at(&self, world_point: Point) -> Vector {
        let object_point = self.trans.inverse() * world_point;
        let object_normal = object_point - point(0.0, 0.0, 0.0);
        let mut world_normal = self.trans.inverse().transpose() * object_normal;
        world_normal.set_w(0.0);
        normalize(world_normal)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LightKind {
    Point,
}

#[derive(Debug, PartialEq)]
struct Light {
    kind: LightKind,
    position: Point,
    intensity: Color,
}

impl Light {
    fn new(kind: LightKind, position: Point, intensity: Color) -> Light {
        Light { kind, position, intensity }
    }
}

fn point_light(position: Point, intensity: Color) -> Light {
    Light::new(LightKind::Point, position, intensity)
}

fn lighting(material: &Material, light: &Light, point: Point, eyev: Vector, normalv: Vector) -> Color {
    let effective_color = hadamard_product(material.color,light.intensity);
    let lightv = normalize(light.position - point);
    let ambient = effective_color * material.ambient;
    let light_dot_normal = dot(lightv, normalv);
    let diffuse;
    let specular;
    let black = color(0.0, 0.0, 0.0);
    if light_dot_normal < 0.0 {
        diffuse = black;
        specular = black;
    } else {
        diffuse = effective_color * material.diffuse * light_dot_normal;
        let reflectv = (-lightv).reflect(normalv);
        let reflect_dot_eye = dot(reflectv, eyev);
        if reflect_dot_eye <= 0.0 {
            specular = black;
        } else {
            let factor = reflect_dot_eye.powf(material.shininess);
            specular = light.intensity * material.specular * factor;
        }
    }
    let mut res = ambient + diffuse + specular;
    res.set_w(1.0);
    res
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Material {
    color: Color,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
}

impl Default for Material {
    fn default() -> Self {
        Material::new(color(1.0, 1.0, 1.0),
                      0.1,
                      0.9,
                      0.9,
                      200.0)
    }
}

impl Material {
    fn new(color: Color, ambient: f32, diffuse: f32, specular: f32, shininess: f32) -> Self {
        Material {
            color,
            ambient,
            diffuse,
            specular,
            shininess
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Intersection {
    t: f32,
    object_id: usize,
}

impl Intersection {
    fn new(t: f32, object_id: usize) -> Intersection {
        Intersection { t, object_id, }
    }
}

fn intersections(is: Vec<Intersection>) -> Vec<Intersection> {
    is
}

fn intersects_at(s: &Shape, ray: &Ray) -> Vec<f32> {
    match s {
        Shape { id: _, prim: Prim::Sphere, .. } => {
            let sphere_to_ray = sub(ray.origin, point(0.0, 0.0, 0.0));
            let a = dot(ray.direction, ray.direction);
            let b = 2.0 * dot(ray.direction, sphere_to_ray);
            let c = dot(sphere_to_ray, sphere_to_ray) - 1.0;
            let discriminant = (b * b) - 4.0 * a * c;
            if discriminant < 0.0 {
                return vec![];
            }
            let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
            let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
            vec![t1, t2]
        },
        _ =>
            vec![]
    }
}

fn intersect(s: &Shape, ray0: &Ray) -> Vec<Intersection> {
    let ray = ray0.transform(&s.trans.inverse());
    match s {
        Shape { id: _, prim: Prim::Sphere, ..} => {
            let sphere_to_ray = sub(ray.origin, point(0.0, 0.0, 0.0));
            let a = dot(ray.direction, ray.direction);
            let b = 2.0 * dot(ray.direction, sphere_to_ray);
            let c = dot(sphere_to_ray, sphere_to_ray) - 1.0;
            let discriminant = (b * b) - 4.0 * a * c;
            if discriminant < 0.0 {
                return vec![];
            }
            let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
            let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
            vec![Intersection::new(t1, s.id), Intersection::new(t2, s.id)]
        },
        _ =>
            vec![]
    }
}

fn hit(is: Vec<Intersection>) -> Option<Intersection> {
    let mut result = None;
    let mut closest_t= f32::MAX;
    for intersection in is {
        // dbg!(intersection.t);
        if intersection.t >= 0.0 {
            if intersection.t < closest_t {
                closest_t = intersection.t;
                result = Some(intersection);
            }
        }
    }
    result
}
struct Env {
    gravity: Vector,
    wind: Vector,
}

impl Env {
    fn new(gravity: Vector, wind: Vector) -> Env {
        Env {gravity, wind}
    }
}

struct Projectile {
    position: Point,
    velocity: Vector,
}

impl Projectile {
    fn new(position: Point, velocity: Point) -> Projectile {
        Projectile {position, velocity}
    }
}

fn tick(env: &Env, proj: &Projectile) -> Projectile {
    let position = add(proj.position, proj.velocity);
    let velocity = add(add(proj.velocity, env.gravity), env.wind);
    Projectile::new(position, velocity)
}

#[allow(dead_code)]
fn picture1() {
    let mut p = Projectile::new(point(0.0, 1.0, 0.0),
                                mul(normalize(vector(1.0, 1.8, 0.0)), 11.25));
    let e = Env::new(vector(0.0, -0.1, 0.0), vector(-0.01, 0.0, 0.0));
    let mut c = Canvas::new(900, 550);
    let col = color(1.0, 0.0, 0.0);
    while p.position.y() > 0.0 {
        println!("position: {:?}, velocity: {:?}", p.position, p.velocity);
        c.write_pixel(p.position.x() as usize, c.height - p.position.y() as usize, col);
        p = tick(&e, &p);
    }
    let ppm = c.to_ppm();
    std::fs::write("boom.ppm", &ppm.content).expect("writing to file");
}

#[allow(dead_code)]
fn picture2() {
    let col = color(1.0, 0.0, 0.0);
    let mut c = Canvas::new(400, 400);

    let twelve = point(0.0, 0.0, 1.0);
    let scale = scaling(130.0, 0.0, 130.0);
    let center = translation(200.0, 0.0, 200.0);

    for i in 0..12 {
        let r = center * rotation_y(i as f32 * std::f32::consts::PI / 6.0) *
            scale;
        let d = r * twelve;
        c.write_pixel(d.x() as usize, d.z() as usize, col);
    }
    let ppm = c.to_ppm();
    std::fs::write("clock.ppm", &ppm.content).expect("writing to file");
}

#[allow(dead_code)]
fn picture3() {
    let ray_origin = point(0.0, 0.0, -5.0);
    let wall_z = 10.0f32;
    let wall_size = 7.0f32;
    let canvas_pixels = 100;
    let pixel_size = wall_size / canvas_pixels as f32;
    let half = wall_size / 2.0;
    let mut canvas = Canvas::new(canvas_pixels, canvas_pixels);
    let circ_color = color(1.0, 0.0, 0.0);
    let shape = sphere(0);

    let sample_size = 50;
    let mut rng = thread_rng();

    for y in 0..canvas_pixels - 1 {
        if y % 10 == 0 {
            println!("Progress: {}%", y as f32 * 100.0 / canvas_pixels as f32);
        }
        for x in 0..canvas_pixels {
            let mut col = color(0.0, 0.0, 0.0);
            for _ in 0..sample_size {
                let delta_x = rng.gen::<f32>() * 2.0 - 1.0;
                let delta_y = rng.gen::<f32>() * 2.0 - 1.0;
                let world_x = -half + pixel_size * (x as f32 + delta_x);
                let world_y = half - pixel_size * (y as f32 + delta_y);
                let position = point(world_x, world_y , wall_z);
                let r = Ray::new(ray_origin, normalize(position - ray_origin));
                let xs = intersect(&shape, &r);
                if let Some(_i) = hit(xs) {
                    col = col + circ_color;
                }
            }
            canvas.write_pixel(x, y, col / sample_size as f32);
        }
    }
    let ppm = canvas.to_ppm();
    std::fs::write("circle.ppm", &ppm.content).expect("writing to file");
}

#[allow(dead_code)]
fn picture4() {
    let ray_origin = point(0.0, 0.0, -5.0);
    let wall_z = 10.0f32;
    let wall_size = 7.0f32;
    let canvas_pixels = 200;
    let pixel_size = wall_size / canvas_pixels as f32;
    let half = wall_size / 2.0;
    let mut canvas = Canvas::new(canvas_pixels, canvas_pixels);
    let mut shape = sphere(0);
    shape.set_material(Material::default());
    shape.mat.color = color(1.0, 0.2, 1.0);
    // shape.set_transform(rotation_z(std::f32::consts::FRAC_PI_2) * scaling(0.8, 0.2, 0.8));

    let light_position = point(-10.0, 10.0, -10.0);
    let light_color = color(1.0, 1.0, 1.0);
    let light = point_light(light_position, light_color);

    let sample_size = 50;
    let mut rng = thread_rng();

    for y in 0..canvas_pixels - 1 {
        if y % 10 == 0 {
            println!("Progress: {}%", y as f32 * 100.0 / canvas_pixels as f32);
        }
        for x in 0..canvas_pixels {
            let mut col = color(0.0, 0.0, 0.0);
            for _ in 0..sample_size {
                let delta_x = rng.gen::<f32>() * 2.0 - 1.0;
                let delta_y = rng.gen::<f32>() * 2.0 - 1.0;
                let world_x = -half + pixel_size * (x as f32 + delta_x);
                let world_y = half - pixel_size * (y as f32 + delta_y);
                let position = point(world_x, world_y , wall_z);
                let r = Ray::new(ray_origin, normalize(position - ray_origin));
                let xs = intersect(&shape, &r);
                if let Some(hit) = hit(xs) {
                    let point = r.position(hit.t);
                    let normal = shape.normal_at(point);
                    let eye = -r.direction;
                    col = col + lighting(shape.material(), &light, point, eye, normal);
                }
            }
            canvas.write_pixel(x, y, col / sample_size as f32);
        }
    }
    let ppm = canvas.to_ppm();
    std::fs::write("circle.ppm", &ppm.content).expect("writing to file");
}

enum R2CMessage {
    Ready { id: usize },
    Result { id: usize, from_row: usize, data: Vec<Vec<Color>> },
}

enum C2RMessage {
    Render { from_row: usize, count: usize },
    Quit,
}

fn render_row(_id: usize, world: Arc<World>, settings: &RenderSettings, samples_per_pixel: usize,
              thread_rng: &mut ThreadRng,
              from_row: usize, count: usize) -> Vec<Vec<Color>> {
    let mut result = Vec::with_capacity(count);
    for y in from_row..(from_row + count) {
        let mut data = Vec::with_capacity(settings.canvas_pixels);
        for x in 0..settings.canvas_pixels {
            let mut col = color(0.0, 0.0, 0.0);
            for _ in 0..samples_per_pixel {
                let delta_x =
                    if samples_per_pixel > 1 {
                        thread_rng.gen::<f32>() * 2.0 - 1.0
                    } else {
                        0.0
                    };
                let delta_y =
                    if samples_per_pixel> 1 {
                        thread_rng.gen::<f32>() * 2.0 - 1.0
                    } else {
                        0.0
                    };
                let world_x = -settings.half + settings.pixel_size * (x as f32 + delta_x);
                let world_y = settings.half - settings.pixel_size * (y as f32 + delta_y);
                let position = point(world_x, world_y, settings.wall_z);
                let r = Ray::new(world.ray_origin, normalize(position - world.ray_origin));
                let mut xs = vec![];
                let mut xs0 = intersect(world.objects.get(&0).unwrap(), &r);
                let mut xs1 = intersect(world.objects.get(&1).unwrap(), &r);
                xs.append(&mut xs0);
                xs.append(&mut xs1);
                let xs = intersections(xs);
                if let Some(hit) = hit(xs) {
                    let point = r.position(hit.t);
                    let shape = world.objects.get(&hit.object_id).unwrap();
                    let normal = shape.normal_at(point);
                    let eye = -r.direction;
                    for light in &world.lights {
                        col = col + lighting(shape.material(), light, point, eye, normal);
                    }
                }
            }
            let c = col / samples_per_pixel as f32;

            data.push(c);
        }
        result.push(data);
    }
    result
}

fn render_thread(id: usize, world: Arc<World>, settings: Arc<RenderSettings>,
                 state: Arc<Mutex<RenderState>>,
                 render_to_control: Sender<R2CMessage>,
                 control_to_render: Receiver<C2RMessage>) {
    let mut thread_rng = rand::thread_rng();

    println!("[render:{}] starting up...", id);
    render_to_control.send(R2CMessage::Ready { id }).expect("sending to control");
    println!("[render:{}] starting up... done.", id);
    loop {
        match control_to_render.recv() {
            Ok(C2RMessage::Render{from_row, count }) => {
                let samples_per_pixel = {
                    let s = state.lock().expect("locking state");
                    s.samples_per_pixel
                };
                println!("[render:{}] rendering row {}..{} (spp: {})...", id, from_row, from_row + count - 1, samples_per_pixel);
                let data = render_row(id, world.clone(), &settings, samples_per_pixel,
                                      &mut thread_rng, from_row, count);
                render_to_control.send(R2CMessage::Result {id, from_row, data}).expect("sending to control thread");
                println!("[render:{}] rendering row {}..{} (spp: {})... done.", id, from_row, from_row + count - 1, samples_per_pixel);
            },
            Ok(C2RMessage::Quit) => break,
            _ =>
                println!("[render:{}] error while receiving message from control", id),
        }
    }
    println!("[render:{}] terminated.", id);
}

struct Computations {
    t: f32,
    object_id: usize,
    point: Point,
    eyev: Vector,
    normalv: Vector,
    inside: bool,
}

fn prepare_computations(w: &World, i: Intersection, ray: &Ray) -> Computations {
    let s = w.objects.get(&i.object_id).expect("getting object");
    let t = i.t;
    let object_id = i.object_id;
    let point = ray.position(t);
    let eyev = -ray.direction;
    let mut normalv = s.normal_at(point);
    let mut inside = false;

    if dot(normalv, eyev) < 0.0 {
        inside = true;
        normalv = -normalv;
    }
    let comps = Computations {
        t,
        object_id,
        point,
        eyev,
        normalv,
        inside,
    };
    comps
}
struct World {
    ray_origin: Point,
    objects: HashMap<usize, Shape>,
    lights: Vec<Light>,
}

impl World {
    fn intersect(&self, ray: &Ray) -> Vec<Intersection> {
        let mut is = Vec::new();
        for s in self.objects.values() {
            let mut i = intersect(s, ray);
            is.append(&mut i);
        }
        is.sort_by(|&i1, &i2| i1.t.partial_cmp(&i2.t).unwrap_or(std::cmp::Ordering::Less));
        is
    }
}

impl Default for World {
    fn default() -> World {
        World {
            ray_origin: point(0.0, 0.0, -10.0),
            objects: HashMap::new(),
            lights: vec![],
        }
    }
}
fn default_test_world() -> World {
    let mut w = World::default();
    w.lights.push(point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0)));
    let mut s = sphere(0);
    s.mat.color = color(0.8, 1.0, 0.6);
    s.mat.diffuse = 0.7;
    s.mat.specular = 0.2;

    w.objects.insert(s.id, s);
    s = sphere(1);
    s.set_transform(scaling(0.5, 0.5, 0.5));
    w.objects.insert(s.id, s);
    w
}

#[derive(Clone)]
struct RenderSettings {
    wall_z: f32,
    canvas_pixels: usize,
    pixel_size: f32,
    half: f32,
    max_samples_per_pixels: usize,
    rows_per_chunk: usize,
}

struct RenderState {
    samples_per_pixel: usize,
}

struct Raytracer {
    #[allow(dead_code)]
    world: Arc<World>,
    settings: Arc<RenderSettings>,
    state: Arc<Mutex<RenderState>>,
    next_row: usize,
    pixel_data: Vec<u8>,
    thread_cnt: usize,
    thread_handles: Vec<JoinHandle<()>>,
    render_to_control_recv: Receiver<R2CMessage>,
    control_to_render_send: Vec<Sender<C2RMessage>>,
}

impl Raytracer {
    pub fn new(_ctx: &mut Context) -> Raytracer {
        let mut objects = HashMap::new();
        let mut shape0 = sphere(0);
        shape0.set_material(Material::default(  ));
        shape0.mat.color = color(1.0, 0.2, 1.0);
        shape0.set_transform(rotation_z(std::f32::consts::PI / 9.0) *
            rotation_y(std::f32::consts::PI / 3.0) *
            scaling(1.5, 0.2, 0.2));
        objects.insert(shape0.id, shape0);

        let mut shape1 = sphere(1);
        shape1.set_material(Material::default());
        shape1.mat.color = color(1.0, 0.2, 0.1);
        shape1.mat.specular = 0.1;
        shape1.mat.shininess = 10.0;
        shape1.set_transform(scaling(0.5, 0.5, 0.5));
        objects.insert(shape1.id, shape1);

        let wall_z= 10.0f32;
        let wall_size = 7.0f32;
        let canvas_pixels = 400;

        let mut lights = Vec::new();
        let light_position = point(-10.0, 10.0, -10.0);
        let light_color = color(0.8, 0.8, 0.8);
        lights.push(point_light(light_position, light_color));

        let light_position = point(10.0, 20.0, -10.0);
        let light_color = color(0.0, 0.05, 0.95);
        lights.push(point_light(light_position, light_color));

        let pixel_data = vec![0; canvas_pixels * canvas_pixels * 4];

        let settings = RenderSettings {
            wall_z,
            pixel_size: wall_size / canvas_pixels as f32,
            half: wall_size / 2.0,
            canvas_pixels,
            max_samples_per_pixels: 200,
            rows_per_chunk: 20,
        };

        let state = RenderState {
          samples_per_pixel: 1,
        };

        let world = World {
            ray_origin: point(0.0, 0.0, -5.0),
            objects,
            lights,
        };

        let world = Arc::new(world);
        let settings = Arc::new(settings);
        let state = Arc::new(Mutex::new(state));

        let thread_cnt = 3;
        let (render_to_control_send, render_to_control_recv) = channel();
        let mut control_to_render = Vec::with_capacity(thread_cnt);
        let mut thread_handles = Vec::with_capacity(thread_cnt);
        for i in 0..thread_cnt {
            let (control_to_render_send, control_to_render_recv) = channel();
            control_to_render.push(control_to_render_send);
            let snd = render_to_control_send.clone();
            let st = settings.clone();
            let stat = state.clone();
            let w = world.clone();
            let id = i.clone();
            let handle = std::thread::spawn(move ||
                render_thread(id, w, st, stat, snd, control_to_render_recv)
            );
            thread_handles.push(handle);
        }

        // Load/create resources such as images here.
        Raytracer {
            world,
            settings,
            state,
            next_row: 0,
            pixel_data,
            thread_handles,
            thread_cnt,
            control_to_render_send: control_to_render,
            render_to_control_recv,
        }
    }
}

impl EventHandler for Raytracer {
    fn update(&mut self, _ctx: &mut Context) -> GameResult<()> {
        for _ in 0..20 {
            let msg = self.render_to_control_recv.try_recv();
            let samples_per_pixel = {
                let state = self.state.lock().expect("locking mutex");
                state.samples_per_pixel
            };
            match msg {
                Ok(R2CMessage::Ready { id }) => {
                    println!("Render thread {} reports ready.", id);
                    if self.next_row < self.settings.canvas_pixels {
                        self.send_chunk(id);
                    }
                }
                Ok(R2CMessage::Result { id, from_row, data }) => {
                    for (y, row_data) in (from_row..(from_row + data.len())).zip(data.iter()) {
                        for x in 0..self.settings.canvas_pixels {
                            let c = row_data[x];
                            let r = component_to_ppm(c.red());
                            let g = component_to_ppm(c.green());
                            let b = component_to_ppm(c.blue());

                            self.pixel_data[(self.settings.canvas_pixels * y + x) * 4 + 0] = r;
                            self.pixel_data[(self.settings.canvas_pixels * y + x) * 4 + 1] = g;
                            self.pixel_data[(self.settings.canvas_pixels * y + x) * 4 + 2] = b;
                            self.pixel_data[(self.settings.canvas_pixels * y + x) * 4 + 3] = 255;
                        }
                    }
                    if self.next_row < self.settings.canvas_pixels {
                        self.send_chunk(id);
                    } else if samples_per_pixel < self.settings.max_samples_per_pixels {
                        {
                            let mut state = self.state.lock().expect("locking mutex");
                            let s = state.samples_per_pixel + 20;
                            state.samples_per_pixel =  s - s % 20;
                        }
                        self.next_row = 0;
                        self.send_chunk(id);
                    }
                },
                _ => { return Ok(()); }
            }
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, GgezColor::WHITE);

        let img = graphics::Image::from_rgba8(ctx,
                                              self.settings.canvas_pixels as u16,
                                              self.settings.canvas_pixels as u16,
                                              &self.pixel_data)?;
        let (_w, h) = graphics::drawable_size(ctx);
        graphics::draw(ctx, &img, graphics::DrawParam::default()
            .scale(glam::vec2(h / self.settings.canvas_pixels as f32,
                              h / self.settings.canvas_pixels as f32)))?;
        graphics::present(ctx)
    }

    fn key_down_event(&mut self, ctx: &mut Context, keycode: ggez::event::KeyCode, _keymods: KeyMods, _repeat: bool) {
        if keycode == ggez::event::KeyCode::Escape {
            self.quit_action();
            ggez::event::quit(ctx);
        }
    }

    fn quit_event(&mut self, _ctx: &mut Context) -> bool {
        self.quit_action();
        false
    }
}

impl Raytracer {
    fn send_chunk(&mut self, id: usize) {
        let rows = std::cmp::min(self.settings.canvas_pixels - self.next_row, self.settings.rows_per_chunk);
        self.control_to_render_send[id]
            .send(C2RMessage::Render { from_row: self.next_row, count: rows })
            .expect("sending to render thread");
        self.next_row += rows;
    }

    fn quit_action(&mut self) {
        println!("[control] notifying render threads to quit...");
        for id in 0..self.thread_cnt {
            self.control_to_render_send[id].send(C2RMessage::Quit).expect("sending to render thread");
        }
        println!("[control] waiting for render threads to finish...");
        for h in self.thread_handles.drain(..) {
            h.join().expect("joining thread");
        }
        println!("[control] done.");

    }
}
fn main() {
    // Make a Context.
    let (mut ctx, event_loop) = ContextBuilder::new("ray_tracer_challenge", "DMG")
        .build()
        .expect("aieee, could not create ggez context!");

    let window_size = 900.0;
    graphics::set_drawable_size(&mut ctx, window_size, window_size)
        .expect("setting drawable size");
    graphics::set_screen_coordinates(&mut ctx, Rect{x: 0.0, y: 0.0, w: window_size, h: window_size})
        .expect("setting screen coordinates");
    // Create an instance of your event handler.
    // Usually, you should provide it with the Context object to
    // use when setting your game up.
    let tracer = Raytracer::new(&mut ctx);

    // Run!
    event::run(ctx, event_loop, tracer);

//    picture4();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn w_one_is_point() {
        let a = Tuple::new(4.3, -4.2, 3.1, 1.0);
        assert!(a.is_point());
        assert!(!a.is_vector());
        assert_eq!(a.x(), 4.3);
        assert_eq!(a.y(), -4.2);
        assert_eq!(a.z(), 3.1);
    }

    #[test]
    fn w_zero_is_vector() {
        let a = Tuple::new(4.3, -4.2, 3.1, 0.0);
        assert!(!a.is_point());
        assert!(a.is_vector());
        assert_eq!(a.x(), 4.3);
        assert_eq!(a.y(), -4.2);
        assert_eq!(a.z(), 3.1);
    }

    #[test]
    fn create_point() {
        let p = point(4.0, -4.0, 3.0);
        assert_eq!(p, Tuple::new(4.0, -4.0, 3.0, 1.0));
    }

    #[test]
    fn create_vector() {
        let v = vector(4.0, -4.0, 3.0);
        assert_eq!(v, Tuple::new(4.0, -4.0, 3.0, 0.0));
    }

    #[test]
    fn add_tuples() {
        let a1 = tuple(3.0, -2.0, 5.0, 1.0);
        let a2 = tuple(-2.0, 3.0, 1.0, 0.0);
        assert_eq!(add(a1, a2), tuple(1.0, 1.0, 6.0, 1.0));
    }

    #[test]
    fn sub_two_points() {
        let p1 = point(3.0, 2.0, 1.0);
        let p2 = point(5.0, 6.0, 7.0);
        assert_eq!(sub(p1, p2), vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_vector_from_point() {
        let p = point(3.0, 2.0, 1.0);
        let v = vector(5.0, 6.0, 7.0);
        assert_eq!(sub(p, v), point(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_two_vectors() {
        let v1 = vector(3.0, 2.0, 1.0);
        let v2 = vector(5.0, 6.0, 7.0);
        assert_eq!(sub(v1, v2), vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_from_zero_negates() {
        let z = zero();
        let v = vector(1.0, -2.0, 3.0);
        assert_eq!(sub(z, v), vector(-1.0, 2.0, -3.0));
    }

    #[test]
    fn negate_tuple() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(negate(a), tuple(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn mul_tuple_by_scalar() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(mul(a, 3.5), tuple(3.5, -7.0, 10.5, -14.0))
    }

    #[test]
    fn mul_tuple_by_fraction() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(mul(a, 0.5), tuple(0.5, -1.0, 1.5, -2.0))
    }

    #[test]
    fn div_tuple_by_scalar() {
        let a = tuple(1.0, -2.0, 3.0, -4.0);
        assert_eq!(div(a, 2.0), tuple(0.5, -1.0, 1.5, -2.0))
    }

    #[test]
    fn vector_magnitude_x() {
        let v = vector(1.0, 0.0, 0.0);
        assert_eq!(magnitude(v), 1.0);
    }

    #[test]
    fn vector_magnitude_y() {
        let v = vector(0.0, 1.0, 0.0);
        assert_eq!(magnitude(v), 1.0);
    }

    #[test]
    fn vector_magnitude_z() {
        let v = vector(0.0, 0.0, 1.0);
        assert_eq!(magnitude(v), 1.0);
    }

    #[test]
    fn vector_magnitude_1() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(magnitude(v), 14f32.sqrt());
    }

    #[test]
    fn vector_magnitude_2() {
        let v = vector(-1.0, -2.0, -3.0);
        assert_eq!(magnitude(v), 14f32.sqrt());
    }

    #[test]
    fn normalize_vec_1() {
        let v = vector(4.0, 0.0, 0.0);
        assert_eq!(normalize(v), vector(1.0, 0.0, 0.0));
    }

    #[test]
    fn normalize_vec_2() {
        let v = vector(1.0, 2.0, 3.0);
        assert_eq!(normalize(v), vector(0.26726124, 0.5345225, 0.8017837));
    }

    #[test]
    fn normalized_magnitude() {
        let v = vector(1.0, 2.0, 3.0);
        let norm = normalize(v);
        assert!(approx(magnitude(norm), 1.0));
    }

    #[test]
    fn dot_product() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);
        assert_eq!(dot(a, b), 20.0);
    }

    #[test]
    fn cross_product() {
        let a = vector(1.0, 2.0, 3.0);
        let b = vector(2.0, 3.0, 4.0);
        assert_eq!(cross(a, b), vector(-1.0, 2.0, -1.0));
        assert_eq!(cross(b, a), vector(1.0, -2.0, 1.0));
    }

    #[test]
    fn colors_are_tuples() {
        let c = color(-0.5, 0.4, 1.7);
        assert_eq!(c.red(), -0.5);
        assert_eq!(c.green(), 0.4);
        assert_eq!(c.blue(), 1.7);
    }

    #[test]
    fn adding_colors() {
        let c1 = color(0.9, 0.6, 0.75);
        let c2 = color(0.7, 0.1, 0.25);
        assert_eq!(add(c1, c2), tuple(1.5999999, 0.70000005, 1.0, 2.0))
    }

    #[test]
    fn subtracting_colors() {
        let c1 = color(0.9, 0.6, 0.75);
        let c2 = color(0.7, 0.1, 0.25);
        assert_eq!(sub(c1, c2), tuple(0.19999999, 0.5, 0.5, 0.0));
    }

    #[test]
    fn multiplying_color_by_scalar() {
        let c = color(0.2, 0.3, 0.4);
        assert_eq!(mul(c, 2.0), tuple(0.4, 0.6, 0.8, 2.0));
    }

    #[test]
    fn multiplying_colors() {
        let c1 = color(1.0, 0.2, 0.4);
        let c2 = color(0.9, 1.0, 0.1);
        assert_eq!(hadamard_product(c1, c2), color(0.9, 0.2, 0.040000003));
    }

    #[test]
    fn create_canvas() {
        let c = Canvas::new(10, 20);
        assert_eq!(c.width, 10);
        assert_eq!(c.height, 20);
        for j in 0..c.height {
            for i in 0..c.width {
                assert_eq!(c.pixel_at(i, j), color(0.0, 0.0, 0.0));
            }
        }
    }

    #[test]
    fn write_pixels() {
        let mut c = Canvas::new(10, 20);
        let red = color(1.0, 0.0, 0.0);
        c.write_pixel(2, 3, red);
        assert_eq!(c.pixel_at(2, 3), red);
    }

    #[test]
    fn constructing_ppm_header() {
        let c = Canvas::new(5, 3);
        let ppm = c.to_ppm();
        assert!(String::from_utf8(ppm.content).unwrap().starts_with("P3\n5 3\n255\n"))
    }

    #[test]
    fn constructing_ppm_pixel_data() {
        let mut c = Canvas::new(5, 3);
        let c1 = color(1.5, 0.0, 0.0);
        let c2 = color(0.0, 0.5, 0.0);
        let c3 = color(-0.5, 0.0, 1.0);
        c.write_pixel(0, 0, c1);
        c.write_pixel(2, 1, c2);
        c.write_pixel(4, 2, c3);
        let ppm = c.to_ppm();
        assert_eq!(String::from_utf8_lossy(&ppm.content[..]),
            "P3\n5 3\n255\n\
            255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
            0 0 0 0 0 0 0 127 0 0 0 0 0 0 0\n\
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 255\n");
    }

    #[test]
    fn construct_inspect_4x4_matrix() {
        let m = Matrix::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            5.5, 6.5, 7.5, 8.5,
            9.0, 10.0, 11.0, 12.0,
            13.5, 14.5, 15.5, 16.5
        ]);
        assert_eq!(m.at(0,0), 1.0);
        assert_eq!(m.at(0,3), 4.0);
        assert_eq!(m.at(1,2), 7.5);
        assert_eq!(m.at(2,2), 11.0);
        assert_eq!(m.at(3,0), 13.5);
        assert_eq!(m.at(3,2), 15.5);
    }

    #[test]
    fn construct_inspect_2x2_matrix() {
        let m = Matrix::from_slice(&[
            -3.0, 5.0,
            1.0, -2.0,
        ]);
        assert_eq!(m.at(0,0), -3.0);
        assert_eq!(m.at(0,1), 5.0);
        assert_eq!(m.at(1,0), 1.0);
        assert_eq!(m.at(1,1), -2.0);
    }

    #[test]
    fn construct_inspect_3x3_matrix() {
        let m = Matrix::from_slice( &[
            -3.0, 5.0, 0.0,
            1.0, -2.0, -7.0,
            0.0, 1.0, 1.0,
        ]);
        assert_eq!(m.at(1,1), -2.0);
        assert_eq!(m.at(2,2), 1.0);
    }

    #[test]
    fn matrix_equality() {
        let a = Matrix::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        let b = Matrix::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]);
        assert_eq!(a, b);
    }

    #[test]
    fn matrix_inequality() {
        let a = Matrix::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 8.0, 7.0, 6.0,
            5.0, 4.0, 3.0, 2.0,
        ]);
        let b = Matrix::from_slice(&[
            2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0,
            8.0, 7.0, 6.0, 5.0,
            4.0, 3.0, 2.0, 1.0,
        ]);
        assert_ne!(a, b);
    }

    #[test]
    fn multiplying_two_matrices() {
        let a = Matrix::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 8.0, 7.0, 6.0,
            5.0, 4.0, 3.0, 2.0,
        ]);
        let b = Matrix::from_slice(&[
            -2.0, 1.0, 2.0, 3.0,
            3.0, 2.0, 1.0, -1.0,
            4.0, 3.0, 6.0, 5.0,
            1.0, 2.0, 7.0, 8.0,
        ]);
        let c = Matrix::from_slice(&[
            20.0, 22.0, 50.0, 48.0,
            44.0, 54.0, 114.0, 108.0,
            40.0, 58.0, 110.0, 102.0,
            16.0, 26.0, 46.0, 42.0,
        ]);
        assert_eq!(a * b, c);
    }

    #[test]
    fn multiply_matrix_and_tuple() {
        let a = Matrix::from_slice(&[
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 4.0, 2.0,
            8.0, 6.0, 4.0, 1.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let b = tuple(1.0, 2.0, 3.0, 1.0);
        assert_eq!(a * b, tuple(18.0, 24.0, 33.0, 1.0));
    }

    #[test]
    fn multiply_identity() {
        let a = Matrix::from_slice(&[
            0.0, 1.0, 2.0, 4.0,
            1.0, 2.0, 4.0, 8.0,
            2.0, 4.0, 8.0, 16.0,
            4.0, 8.0, 16.0, 32.0,
        ]);
        assert_eq!(a * Matrix::identity(), a);
    }

    #[test]
    fn transposing() {
        let a = Matrix::from_slice(&[
            0.0, 9.0, 3.0, 0.0,
            9.0, 8.0, 0.0, 8.0,
            1.0, 8.0, 5.0, 3.0,
            0.0, 0.0, 5.0, 8.0,
        ]);
        let b = Matrix::from_slice(&[
            0.0, 9.0, 1.0, 0.0,
            9.0, 8.0, 8.0, 0.0,
            3.0, 0.0, 5.0, 5.0,
            0.0, 8.0, 3.0, 8.0,
        ]);
        assert_eq!(a.transpose(), b);
    }

    #[test]
    fn transpose_identity() {
        assert_eq!(Matrix::identity().transpose(), Matrix::identity());
    }

    #[test]
    fn determinant_2x2() {
        let m = Matrix::from_slice(&[
            1.0, 5.0,
            -3.0, 2.0,
        ]);
        assert_eq!(m.determinant(), 17.0);
    }

    #[test]
    fn submatrix_3x3_is_2x2() {
        let a = Matrix::from_slice(&[
            1.0, 5.0, 0.0,
            -3.0, 2.0, 7.0,
            0.0, 6.0, -3.0,
        ]);
        let c = Matrix::from_slice(&[
            -3.0, 2.0,
            0.0, 6.0,
        ]);
        assert_eq!(a.submatrix(0, 2), c);
    }

    #[test]
    fn submatrix_4x4_is_3x3() {
        let a = Matrix::from_slice(&[
            -6.0, 1.0, 1.0, 6.0,
            -8.0, 5.0, 8.0, 6.0,
            -1.0, 0.0, 8.0, 2.0,
            -7.0, 1.0, -1.0, 1.0,
        ]);
        let c = Matrix::from_slice(&[
            -6.0, 1.0, 6.0,
            -8.0, 8.0, 6.0,
            -7.0, -1.0, 1.0,
        ]);
        assert_eq!(a.submatrix(2, 1), c);
    }

    #[test]
    fn minor_3x3() {
        let a = Matrix::from_slice(&[
            3.0, 5.0, 0.0,
            2.0, -1.0, -7.0,
            6.0, -1.0, 5.0,
        ]);
        let b = a.submatrix(1, 0);
        assert_eq!(b.determinant(), 25.0);
        assert_eq!(a.minor(1, 0), 25.0);
    }

    #[test]
    fn cofactor() {
        let a = Matrix::from_slice(&[
            3.0, 5.0, 0.0,
            2.0, -1.0, -7.0,
            6.0, -1.0, 5.0,
        ]);
        assert_eq!(a.minor(0,0), -12.0);
        assert_eq!(a.cofactor(0, 0), -12.0);
        assert_eq!(a.minor(1, 0), 25.0);
        assert_eq!(a.cofactor(1, 0), -25.0);
    }

    #[test]
    fn determinant_3x3() {
        let a = Matrix::from_slice(&[
            1.0, 2.0, 6.0,
            -5.0, 8.0, -4.0,
            2.0, 6.0, 4.0,
        ]);
        assert_eq!(a.cofactor(0,0), 56.0);
        assert_eq!(a.cofactor(0, 1), 12.0);
        assert_eq!(a.cofactor(0, 2), -46.0);
        assert_eq!(a.determinant(), -196.0);
    }

    #[test]
    fn determinant_4x4() {
        let a = Matrix::from_slice(&[
            -2.0, -8.0, 3.0, 5.0,
            -3.0, 1.0, 7.0, 3.0,
            1.0, 2.0, -9.0, 6.0,
            -6.0, 7.0, 7.0, -9.0,
        ]);
        assert_eq!(a.cofactor(0,0), 690.0);
        assert_eq!(a.cofactor(0, 1), 447.0);
        assert_eq!(a.cofactor(0, 2), 210.0);
        assert_eq!(a.cofactor(0, 3), 51.0);
        assert_eq!(a.determinant(), -4071.0);
    }

    #[test]
    fn test_invertible() {
        let a = Matrix::from_slice(&[
            6.0, 4.0, 4.0, 4.0,
            5.0, 5.0, 7.0, 6.0,
            4.0, -9.0, 3.0, -7.0,
            9.0, 1.0, 7.0, -6.0,
        ]);
        assert_eq!(a.determinant(), -2120.0);
        assert!(a.invertible());
    }

    #[test]
    fn test_noninvertible() {
        let a = Matrix::from_slice(&[
            -4.0, 2.0, -2.0, -3.0,
            9.0, 6.0, 2.0, 6.0,
            0.0, -5.0, 1.0, -5.0,
            0.0, 0.0, 0.0, 0.0,
        ]);
        assert_eq!(a.determinant(), 0.0);
        assert!(!a.invertible());
    }

    #[test]
    fn invert_matrix() {
        let a = Matrix::from_slice(&[
            -5.0, 2.0, 6.0, -8.0,
            1.0, -5.0, 1.0, 8.0,
            7.0, 7.0, -6.0, -7.0,
            1.0, -3.0, 7.0, 4.0,
        ]);
        let b = a.inverse();
        assert_eq!(a.determinant(), 532.0);
        assert_eq!(a.cofactor(2, 3), -160.0);
        assert_eq!(b.at(3, 2), -160.0 / 532.0);
        assert_eq!(a.cofactor(3, 2), 105.0);
        assert_eq!(b.at(2, 3), 105.0 / 532.0);
        let c = Matrix::from_slice(&[
            0.21804512, 0.45112783, 0.24060151, -0.04511278,
            -0.8082707, -1.456767, -0.44360903, 0.5206767,
            -0.078947365, -0.2236842, -0.05263158, 0.19736843,
            -0.52255636, -0.81390977, -0.30075186, 0.30639097,
        ]);
        assert_eq!(b, c);
    }

    #[test]
    fn invert_other_matrix() {
        let a = Matrix::from_slice(&[
            8.0, -5.0, 9.0, 2.0,
            7.0, 5.0, 6.0, 1.0,
            -6.0, 0.0, 9.0, 6.0,
            -3.0, 0.0, -9.0, -4.0,
        ]);
        let b = Matrix::from_slice(&[
            -0.15384616, -0.15384616, -0.2820513, -0.53846157,
            -0.07692308, 0.12307692, 0.025641026, 0.03076923,
            0.35897437, 0.35897437, 0.43589744, 0.9230769,
            -0.6923077, -0.6923077, -0.7692308, -1.9230769,
            // -0.15385, -0.15385, -0.28205, -0.53846,
            // -0.07692, 0.12308, 0.02564, 0.03077,
            // 0.35897, 0.35897, 0.43590, 0.92308,
            // -0.69231, -0.69231, -0.76923, -1.92308,
        ]);
        assert_eq!(a.inverse(), b);
    }

    #[test]
    fn invert_third_matrix() {
        let a = Matrix::from_slice(&[
            9.0, 3.0, 0.0, 9.0,
            -5.0, -2.0, -6.0, -3.0,
            -4.0, 9.0, 6.0, 4.0,
            -7.0, 6.0, 6.0, 2.0,
        ]);
        let b = Matrix::from_slice(&[
            -0.04074074, -0.07777778, 0.14444445, -0.22222222,
            -0.07777778, 0.033333335, 0.36666667, -0.33333334,
            -0.029012345, -0.14629629, -0.10925926, 0.12962963,
            0.17777778, 0.06666667, -0.26666668, 0.33333334,
        ]);
        assert_eq!(a.inverse(), b);
    }

    #[test]
    fn multiply_by_inverse() {
        let a = Matrix::from_slice(&[
            3.0, -9.0, 7.0, 3.0,
            3.0, -8.0, 2.0, -9.0,
            -4.0, 4.0, 4.0, 1.0,
            -6.0, 5.0, -1.0, 1.0,
        ]);
        let b = Matrix::from_slice(&[
            8.0, 2.0, 2.0, 2.0,
            3.0, -1.0, 7.0, 0.0,
            7.0, 0.0, 5.0, 4.0,
            6.0, -2.0, 0.0, 5.0,
        ]);
        let c = a * b;
        assert_eq!(c * b.inverse(), a);
    }

    #[test]
    fn mul_translation_matrix() {
        let transform = translation(5.0, -3.0, 2.0);
        let p = point(-3.0, 4.0, 5.0);
        assert_eq!(transform * p, point(2.0, 1.0, 7.0));
    }

    #[test]
    fn mul_inv_translation_matrix() {
        let transform = translation(5.0, -3.0, 2.0);
        let inv = transform.inverse();
        let p = point(-3.0, 4.0, 5.0);
        assert_eq!(inv * p, point(-8.0, 7.0, 3.0));
    }

    #[test]
    fn translation_not_affecting_vectors() {
        let transform = translation(5.0, -3.0, 2.0);
        let v = vector(-3.0, 4.0, 5.0);
        assert_eq!(transform * v, v);
    }

    #[test]
    fn scaling_matrix_point() {
        let transform = scaling(2.0, 3.0, 4.0);
        let p = point(-4.0, 6.0, 8.0);
        assert_eq!(transform * p, point(-8.0, 18.0, 32.0));
    }

    #[test]
    fn scaling_matrix_vector() {
        let transform = scaling(2.0, 3.0, 4.0);
        let v = vector(-4.0, 6.0, 8.0);
        assert_eq!(transform * v, vector(-8.0, 18.0, 32.0));
    }

    #[test]
    fn scaling_inverse() {
        let transform = scaling(2.0, 3.0, 4.0);
        let inv = transform.inverse();
        let v = vector(-4.0, 6.0, 8.0);
        assert_eq!(inv * v, vector(-2.0, 2.0, 2.0));
    }

    #[test]
    fn reflection_by_scaling() {
        let transform = scaling(-1.0, 1.0, 1.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(-2.0, 3.0, 4.0));
    }

    #[test]
    fn rotation_around_x() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_x(std::f32::consts::FRAC_PI_4);
        let full_quarter = rotation_x(std::f32::consts::FRAC_PI_2);
        assert_eq!(half_quarter * p, point(0.0, 2.0f32.sqrt() / 2.0, 2.0f32.sqrt() / 2.0));
        assert_eq!(full_quarter * p, point(0.0, 0.0, 1.0));
    }

    #[test]
    fn inv_x_rotation() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_x(std::f32::consts::FRAC_PI_4);
        let inv = half_quarter.inverse();
        assert_eq!(inv * p, point(0.0, 2.0f32.sqrt() / 2.0, -2.0f32.sqrt() / 2.0));
    }

    #[test]
    fn rotation_around_y() {
        let p = point(0.0, 0.0, 1.0);
        let half_quarter = rotation_y(std::f32::consts::FRAC_PI_4);
        let full_quarter = rotation_y(std::f32::consts::FRAC_PI_2);
        assert_eq!(half_quarter * p, point(2.0f32.sqrt() / 2.0, 0.0, 2.0f32.sqrt() / 2.0));
        assert_eq!(full_quarter * p, point(1.0, 0.0, 0.0));
    }

    #[test]
    fn rotation_around_z() {
        let p = point(0.0, 1.0, 0.0);
        let half_quarter = rotation_z(std::f32::consts::FRAC_PI_4);
        let full_quarter = rotation_z(std::f32::consts::FRAC_PI_2);
        assert_eq!(half_quarter * p, point(-2.0f32.sqrt() / 2.0, 2.0f32.sqrt() / 2.0, 0.0));
        assert_eq!(full_quarter * p, point(-1.0, 0.0, 0.0));
    }

    #[test]
    fn shearing_x_to_y() {
        let transform = shearing(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(5.0, 3.0, 4.0));
    }

    #[test]
    fn shearing_x_to_z() {
        let transform = shearing(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(6.0, 3.0, 4.0));
    }

    #[test]
    fn shearing_y_to_x() {
        let transform = shearing(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(2.0, 5.0, 4.0));
    }

    #[test]
    fn shearing_y_to_z() {
        let transform = shearing(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(2.0, 7.0, 4.0));
    }

    #[test]
    fn shearing_z_to_x() {
        let transform = shearing(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(2.0, 3.0, 6.0));
    }

    #[test]
    fn shearing_z_to_y() {
        let transform = shearing(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let p = point(2.0, 3.0, 4.0);
        assert_eq!(transform * p, point(2.0, 3.0, 7.0));
    }

    #[test]
    fn transformations_in_sequence() {
        let p = point(1.0, 0.0, 1.0);
        let a = rotation_x(std::f32::consts::FRAC_PI_2);
        let b = scaling(5.0, 5.0, 5.0);
        let c = translation(10.0, 5.0, 7.0);
        let p2 = a * p;
        assert_eq!(p2, point(1.0, -1.0, 0.0));
        let p3 = b * p2;
        assert_eq!(p3, point(5.0, -5.0, 0.0));
        let p4 = c * p3;
        assert_eq!(p4, point(15.0, 0.0, 7.0));
    }

    #[test]
    fn chained_transformations() {
        let p = point(1.0, 0.0, 1.0);
        let a = rotation_x(std::f32::consts::FRAC_PI_2);
        let b = scaling(5.0, 5.0, 5.0);
        let c = translation(10.0, 5.0, 7.0);
        let t = c * b * a;
        assert_eq!(t * p, point(15.0, 0.0, 7.0));
    }

    #[test]
    fn creating_ray() {
        let origin = point(1.0, 2.0, 3.0);
        let direction = vector(4.0, 5.0, 6.0);
        let r = Ray::new(origin, direction);
        assert_eq!(r.origin, origin);
        assert_eq!(r.direction, direction);
    }

    #[test]
    fn point_from_distance() {
        let r = Ray::new(point(2.0, 3.0, 4.0), vector(1.0, 0.0, 0.0));
        assert_eq!(r.position(0.0), point(2.0, 3.0, 4.0));
        assert_eq!(r.position(1.0), point(3.0, 3.0, 4.0));
        assert_eq!(r.position(-1.0), point(1.0, 3.0, 4.0));
        assert_eq!(r.position(2.5), point(4.5, 3.0, 4.0));
    }

    #[test]
    fn ray_intersects_sphere_two_points() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere(0);
        let xs = intersects_at(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0], 4.0);
        assert_eq!(xs[1], 6.0);
    }

    #[test]
    fn ray_intersects_sphere_at_tangent() {
        let r = Ray::new(point(0.0, 1.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere(0);
        let xs = intersects_at(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0], 5.0);
        assert_eq!(xs[1], 5.0);
    }

    #[test]
    fn ray_misses_sphere() {
        let r = Ray::new(point(0.0, 2.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere(0);
        let xs = intersects_at(&s, &r);
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn ray_originates_in_sphere() {
        let r = Ray::new(point(0.0, 0.0, 0.0), vector(0.0, 0.0, 1.0));
        let s = sphere(0);
        let xs = intersects_at(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0], -1.0);
        assert_eq!(xs[1], 1.0);
    }

    #[test]
    fn sphere_behind_ray() {
        let r = Ray::new(point(0.0, 0.0, 5.0), vector(0.0, 0.0, 1.0));
        let s = sphere(0);
        let xs = intersects_at(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0], -6.0);
        assert_eq!(xs[1], -4.0);
    }

    #[test]
    fn intersection_encapsulates() {
        let s = sphere(1);
        let i = Intersection::new(3.5, s.id);
        assert_eq!(i.t, 3.5);
        assert_eq!(i.object_id, s.id);
    }

    #[test]
    fn aggregating_intersections() {
        let s = sphere(1);
        let i1 = Intersection::new(1.0, s.id);
        let i2 = Intersection::new(2.0, s.id);
        let xs = intersections(vec![i1, i2]);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 1.0);
        assert_eq!(xs[1].t, 2.0);
    }

    #[test]
    fn intersect_sets_object() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere(1);
        let xs = intersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].object_id, s.id);
        assert_eq!(xs[1].object_id, s.id);
    }

    #[test]
    fn hit_with_positive_intersections() {
        let s = sphere(1);
        let i1 = Intersection::new(1.0, s.id);
        let i2 = Intersection::new(2.0, s.id);
        let xs = intersections(vec![i1, i2]);
        let i = hit(xs);
        assert_eq!(i, Some(i1));
    }

    #[test]
    fn hit_with_intersection_neg_t() {
        let s = sphere(1);
        let i1 = Intersection::new(-1.0, s.id);
        let i2 = Intersection::new(1.0, s.id);
        let xs = intersections(vec![i1, i2]);
        let i = hit(xs);
        assert_eq!(i, Some(i2));
    }

    #[test]
    fn hit_with_all_neg_t() {
        let s = sphere(1);
        let i1 = Intersection::new(-2.0, s.id);
        let i2 = Intersection::new(-1.0, s.id);
        let xs = intersections(vec![i1, i2]);
        let i = hit(xs);
        assert_eq!(i, None);
    }

    #[test]
    fn hit_lowest_nonneg_intersection() {
        let s = sphere(0);
        let i1 = Intersection::new(5.0, s.id);
        let i2 = Intersection::new(7.0, s.id);
        let i3 = Intersection::new(-3.0, s.id);
        let i4 = Intersection::new(2.0, s.id);
        let xs = intersections(vec![i1, i2, i3, i4]);
        let i = hit(xs);
        assert_eq!(i, Some(i4));
    }

    #[test]
    fn translating_a_ray() {
        let r = Ray::new(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0));
        let m = translation(3.0, 4.0, 5.0);
        let r2 = r.transform(&m);
        assert_eq!(r2.origin, point(4.0, 6.0, 8.0));
        assert_eq!(r2.direction, vector(0.0, 1.0, 0.0));
    }

    #[test]
    fn scaling_a_ray() {
        let r = Ray::new(point(1.0, 2.0, 3.0), vector(0.0, 1.0, 0.0));
        let m = scaling(2.0, 3.0, 4.0);
        let r2 = r.transform(&m);
        assert_eq!(r2.origin, point(2.0, 6.0, 12.0));
        assert_eq!(r2.direction, vector(0.0, 3.0, 0.0));
    }

    #[test]
    fn spheres_default_transformation() {
        let s = sphere(0);
        assert_eq!(s.transform(), Matrix::identity());
    }

    #[test]
    fn changing_spheres_transformation() {
        let mut s = sphere(0);
        let t = translation(2.0, 3.0, 4.0);
        s.set_transform(t);
        assert_eq!(s.transform(), t);
    }

    #[test]
    fn intersecting_scaled_sphere_with_ray() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0,0.0, 1.0));
        let mut s = sphere(0);
        s.set_transform(scaling(2.0, 2.0, 2.0));
        let xs = intersect(&s, &r);
        assert_eq!(xs.len(), 2);
        assert_eq!(xs[0].t, 3.0);
        assert_eq!(xs[1].t, 7.0);
    }

    #[test]
    fn intersecting_translated_sphere_with_ray() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let mut s = sphere(0);
        s.set_transform(translation(5.0, 0.0, 0.0));
        let xs = intersect(&s, &r);
        assert_eq!(xs.len(), 0);
    }

    #[test]
    fn normal_on_sphere_on_x_axis() {
        let s = sphere(0);
        let n = s.normal_at(point(1.0, 0.0, 0.0));
        assert_eq!(n, vector(1.0, 0.0, 0.0));
    }

    #[test]
    fn normal_on_sphere_on_non_axial_point() {
        let s = sphere(0);
        let c = 3.0f32.sqrt() / 3.0;
        let n = s.normal_at(point(c, c, c));
        assert_eq!(n, vector(c, c, c));
    }

    #[test]
    fn normal_is_normalized() {
        let s = sphere(0);
        let c = 3.0f32.sqrt() / 3.0;
        let n = s.normal_at(point(c, c, c));
        assert_eq!(n, normalize(n));
    }

    #[test]
    fn normal_on_translated_sphere() {
        let mut s = sphere(0);
        s.set_transform(translation(0.0, 1.0, 0.0));
        let n = s.normal_at(point(0.0, 1.70711, -0.70711));
        assert_eq!(n, vector(0.0, 0.7071068, -0.70710677));
    }

    #[test]
    fn normal_on_transformed_sphere() {
        let mut s = sphere(0);
        let m = scaling(1.0, 0.5, 1.0) * rotation_z(std::f32::consts::PI / 5.0);
        s.set_transform(m);
        let n = s.normal_at(point(0.0, 2.0f32.sqrt() / 2.0, -2.0f32.sqrt() / 2.0));
        assert_eq!(n, vector(0.0, 0.97014254, -0.24253564));
    }

    #[test]
    fn reflecting_vector_45_degrees() {
        let v = vector(1.0, -1.0, 0.0);
        let n = vector(0.0, 1.0, 0.0);
        let r = v.reflect(n);
        assert_eq!(r, vector(1.0, 1.0, 0.0));
    }

    #[test]
    fn reflecting_vector_off_slanted_surface() {
        let v = vector(0.0, -1.0, 0.0);
        let n = vector(2.0f32.sqrt() / 2.0, 2.0f32.sqrt() / 2.0, 0.0);
        let r = v.reflect(n);
        assert_eq!(r, vector(1.0, 0.0, 0.0));
    }

    #[test]
    fn point_light_has_position_and_intensity() {
        let intensity = color(1.0, 1.0, 1.0);
        let position = point(0.0, 0.0, 0.0);
        let light = point_light(position, intensity);
        assert_eq!(light.kind, LightKind::Point);
        assert_eq!(light.position, position);
        assert_eq!(light.intensity, intensity);
    }

    #[test]
    fn default_material() {
        let m = Material::default();
        assert_eq!(m.color, color(1.0, 1.0, 1.0));
        assert_eq!(m.ambient, 0.1);
        assert_eq!(m.diffuse, 0.9);
        assert_eq!(m.specular, 0.9);
        assert_eq!(m.shininess, 200.0);
    }

    #[test]
    fn sphere_has_default_material() {
        let s = sphere(0);
        let m = s.material();
        assert_eq!(m, &Material::default());
    }

    #[test]
    fn sphere_may_be_assigned_material() {
        let mut s = sphere(0);
        let mut m = Material::default();
        m.ambient = 1.0;
        s.set_material(m);
        assert_eq!(s.material(), &m);
    }

    fn background() -> (Material, Point) {
        (Material::default(), point(0.0, 0.0, 0.0))
    }

    #[test]
    fn lighting_between_light_and_surface() {
        let (m, position) = background();
        let eyev = vector(0.0, 0.0, -1.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0));
        let result = lighting(&m, &light, position, eyev, normalv);
        assert_eq!(result, color(1.9, 1.9, 1.9));
    }

    #[test]
    fn lighting_between_light_and_surface_eye_offset_45_degrees() {
        let (m, position) = background();
        let eyev = vector(0.0, 2f32.sqrt() / 2.0, -2f32.sqrt() / 2.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, -10.0), color(1.0, 1.0, 1.0));
        let result = lighting(&m, &light, position, eyev, normalv);
        assert_eq!(result, color(1.0, 1.0, 1.0));
    }

    #[test]
    fn lighting_eye_opposite_surface_light_offset_45_degrees() {
        let (m, position) = background();
        let eyev = vector(0.0, 0.0, -1.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 10.0, -10.0), color(1.0, 1.0, 1.0));
        let result = lighting(&m, &light, position, eyev, normalv);
        assert_eq!(result, color(0.7363961, 0.7363961, 0.7363961));
    }

    #[test]
    fn lighting_eye_in_path_of_reflection() {
        let (m, position) = background();
        let eyev = vector(0.0, -(2f32.sqrt() / 2.0), -(2f32.sqrt() / 2.0));
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 10.0, -10.0), color(1.0, 1.0, 1.0));
        let result = lighting(&m, &light, position, eyev, normalv);
        assert_eq!(result, color(1.6363853, 1.6363853, 1.6363853));
    }

    #[test]
    fn lighting_light_behind_surface() {
        let (m, position) = background();
        let eyev = vector(0.0, 0.0, -1.0);
        let normalv = vector(0.0, 0.0, -1.0);
        let light = point_light(point(0.0, 0.0, 10.0), color(1.0, 1.0, 1.0));
        let result = lighting(&m, &light, position, eyev, normalv);
        assert_eq!(result, color(0.1, 0.1, 0.1));
    }

    #[test]
    fn creating_a_world() {
        let w = World::default();
        assert_eq!(w.objects.len(), 0);
        assert_eq!(w.lights.len(), 0);
    }

    #[test]
    fn the_default_world() {
        let light = point_light(point(-10.0, 10.0, -10.0), color(1.0, 1.0, 1.0));
        let mut s1 = sphere(0);
        s1.mat.color = color(0.8, 1.0, 0.6);
        s1.mat.diffuse = 0.7;
        s1.mat.specular = 0.2;
        let mut s2 = sphere(1);
        s2.set_transform(scaling(0.5, 0.5, 0.5));
        let w = default_test_world();
        assert_eq!(w.lights[0], light);
        assert_eq!(w.objects[&0], s1);
        assert_eq!(w.objects[&1], s2);
    }

    #[test]
    fn intersect_world_with_ray() {
        let w = default_test_world();
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let xs = w.intersect(&r);
        assert_eq!(xs.len(), 4);
        assert_eq!(xs[0].t, 4.0);
        assert_eq!(xs[1].t, 4.5);
        assert_eq!(xs[2].t, 5.5);
        assert_eq!(xs[3].t, 6.0);
    }

    #[test]
    fn precomputing_state_of_intersection() {
        let r = Ray::new(point(0.0, 0.0, -5.0), vector(0.0, 0.0, 1.0));
        let s = sphere(0);
        let mut w = World::default();
        w.objects.insert(s.id, s.clone());
        let i = Intersection::new(4.0, s.id);
        let comps = prepare_computations(&w, i, &r);
        assert_eq!(comps.t, i.t);
        assert_eq!(comps.object_id, i.object_id);
        assert_eq!(comps.point, point(0.0, 0.0, -1.0));
        assert_eq!(comps.eyev, vector(0.0, 0.0, -1.0));
        assert_eq!(comps.normalv, vector(0.0, 0.0, -1.0));
    }
}
